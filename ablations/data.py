"""Load full-data stages with chronological 60%, 20%, and 20% splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dmra.runtime.data import cache_metadata, generate_samples, source_shape
from dmra.runtime.data_madrid_shift import generate_madrid_shift_samples, load_madrid_shift_source


DATASETS = ('PEMS', 'AIR', 'CA_NODE_SHIFT',
            'CA_MULTIYEAR_NODE_SHIFT', 'MADRID_2022_2026_MMD')
FULL_PROTOCOL = 'dmra_full_chronological_60_20_20_v1'
_CHECKED_MADRID = set()


def canonical_dataset(value):
    value = str(value).upper().replace('-', '_')
    if value in {'MADRID', 'MADRID_SHIFT'}:
        value = 'MADRID_2022_2026_MMD'
    if value not in DATASETS:
        raise ValueError('DMRA ablation datasets are '+str(DATASETS))
    return value


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root)/path


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(config):
    dataset = canonical_dataset(config['dataset'])
    if config.get('protocol', 'full') != 'full':
        raise ValueError('Expected full-data protocol')
    if config.get('val_test_mix', False):
        raise ValueError('Ablation tracks require chronological validation/test')
    if int(config.get('x_len',12)) != 12 or int(config.get('y_len',12)) != 12:
        raise ValueError('Pinned DMRA comparison uses 12->12 windows')
    rates = tuple(float(config.get(k,d)) for k,d in
                  (('train_rate',.6),('val_rate',.2),('test_rate',.2)))
    if rates != (.6, .2, .2):
        raise ValueError('Pinned data track does not allow split-rate overrides')
    return dataset


def check_sources(root,config):
    """Return missing required assets; reject malformed immutable assets.

    Madrid's source hashes, cohort identities, frozen month-selection assets,
    first-task-only bandwidth and original Gaussian graphs are all validated.
    """
    root=Path(root).resolve();dataset=_validate_config(config)
    begin,end=int(config['begin_year']),int(config['end_year'])
    if end<begin: raise ValueError('end_year must not precede begin_year')
    missing=[]
    for year in range(begin,end+1):
        for directory,suffix in ((config['raw_data_path'],'.npz'),
                                 (config['graph_path'],'_adj.npz')):
            path=_path(root,directory)/(str(year)+suffix)
            if not path.is_file(): missing.append(path)
    if missing: return missing
    if dataset.startswith('MADRID'):
        manifest=_path(root,config['raw_data_path']).parent/'manifest.json'
        if not manifest.is_file(): return [manifest]
        # Validation is read-only; cache only by all declared file stats.  This
        # is an execution optimization, never a substitute for the hash check.
        files=[manifest]
        document=json.loads(manifest.read_text(encoding='utf-8'))
        for row in document.get('tasks',[]):
            for field,directory,suffix in (('raw_file',config['raw_data_path'],'.npz'),
                                           ('graph_file',config['graph_path'],'_adj.npz')):
                files.append(_path(root,directory)/(str(row['year'])+suffix))
        selection=document.get('selection',{})
        for key in ('result_file','protocol_file','anchor_file'):
            if key in selection: files.append(manifest.parent/selection[key])
        if any(not p.is_file() for p in files):
            return [p for p in files if not p.is_file()]
        signature=(str(root),tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files))
        if signature not in _CHECKED_MADRID:
            from dmra.train_madrid import check_data
            validation = dict(config, raw_data_path=str(_path(root, config['raw_data_path'])),
                              graph_path=str(_path(root, config['graph_path'])))
            check_data(validation)
            _CHECKED_MADRID.add(signature)
    for year in range(begin,end+1):
        steps,nodes=source_shape(_path(root,config['raw_data_path'])/f'{year}.npz')
        with np.load(_path(root,config['graph_path'])/f'{year}_adj.npz',allow_pickle=False) as archive:
            graph=np.asarray(archive['x'])
        if graph.shape!=(nodes,nodes) or not np.isfinite(graph).all() or (graph<0).any():
            raise ValueError(f'Invalid physical graph/source dimensions: {dataset}/{year}')
    return []


def load_context(root,config,year,smoke=False):
    root=Path(root).resolve();dataset=_validate_config(config);year=int(year)
    if not int(config['begin_year'])<=year<=int(config['end_year']):
        raise ValueError('Requested task lies outside the configured stage chain')
    source_path=_path(root,config['raw_data_path'])/f'{year}.npz'
    graph_path=_path(root,config['graph_path'])/f'{year}_adj.npz'
    source_digest,graph_source_digest=sha256(source_path),sha256(graph_path)
    with np.load(graph_path,allow_pickle=False) as archive:
        graph=np.array(archive['x'],copy=True)
    if dataset.startswith('MADRID'):
        source=load_madrid_shift_source(source_path)
        values=source['x'];node_ids=source['node_ids'];active=source['active_nodes']
        stop=int(len(values)*.6)
        if not np.array_equal(active,source['valid'][:stop].any(0)):
            raise ValueError('Madrid availability differs from the full training budget')
        inputs=generate_madrid_shift_samples(values,source['valid'],source['timestamps'],
            graph=graph,node_ids=node_ids,active_nodes=active,coordinates=source['coordinates'],
            dst_invalid_clock=source['dst_invalid_clock'],train_rate=.6,val_rate=.2,test_rate=.2,
            x_len=12,y_len=12,normalize_x=bool(config.get('normalize_x',False)),
            normalization_scope=config.get('normalization_scope','train'),
            sample_limits={'train':4,'val':2,'test':2} if smoke else None,
            savepath=None,source_sha256=source_digest)
        source_valid=source['valid']
        from dmra.runtime.data_madrid_shift import _forward_fill
        fitted,_=_forward_fill(values,source_valid,stop)
        train_raw=np.array(fitted[:stop],copy=True)
        manifest_path_full=source_path.parent.parent/'manifest.json'
        manifest_digest=sha256(manifest_path_full)
    else:
        with np.load(source_path,allow_pickle=False) as archive:
            values=np.array(archive['x'],copy=True)
        if dataset=='PEMS' and int(config.get('pems_days',31))>0:
            values=values[:int(config.get('pems_days',31))*int(config.get('steps_per_day',288))]
        node_ids=np.arange(values.shape[1],dtype=np.int64);active=np.ones(values.shape[1],dtype=bool)
        stop=int(len(values)*.6);source_valid=np.isfinite(values)
        train_raw=np.asarray(values[:stop],dtype=np.float32).copy()
        inputs=generate_samples(31,None,values,graph,train_rate=.6,val_rate=.2,test_rate=.2,
            val_test_mix=False,normalize_x=bool(config.get('normalize_x',False)),
            normalization_scope=config.get('normalization_scope','split'),x_len=12,y_len=12,
            sample_limits={'train':4,'val':2,'test':2} if smoke else None)
        manifest_digest=hashlib.sha256(_json(dict(protocol=FULL_PROTOCOL,dataset=dataset,
            rates=[.6,.2,.2],x_len=12,y_len=12,source_rows=len(values))).encode()).hexdigest()
    if graph.shape!=(len(node_ids),len(node_ids)) or not np.isfinite(graph).all() or (graph<0).any():
        raise ValueError('Physical graph does not match the current source registry')
    metadata=cache_metadata(inputs)
    graph_digest=hashlib.sha256(graph.tobytes(order='C')).hexdigest()
    metadata.update(protocol=FULL_PROTOCOL,dataset=dataset,year=year,train_allowed_stop=stop,
        source_sha256=source_digest,graph_source_sha256=graph_source_digest,
        graph_sha256=graph_digest,node_ids_sha256=hashlib.sha256(node_ids.tobytes()).hexdigest(),
        active_nodes_sha256=hashlib.sha256(active.tobytes()).hexdigest(),
        manifest_sha256=manifest_digest,training_fraction_of_entire_task=.6,
        formal_source_steps=len(values),node_identity='stable_madrid_registry' if dataset.startswith('MADRID') else 'legacy_processed_column_prefix')
    semantic={k:metadata[k] for k in ('protocol','dataset','year','source_sha256','graph_source_sha256',
        'graph_sha256','node_ids_sha256','active_nodes_sha256','train_allowed_stop','normalize_x','normalization_scope')}
    plan_digest=hashlib.sha256(_json(semantic).encode()).hexdigest()
    metadata['plan_sha256']=plan_digest
    inputs['__metadata__']=np.asarray(_json(metadata))
    inputs['node_ids']=node_ids;inputs['active_nodes']=active
    return SimpleNamespace(inputs=inputs,raw_adj=graph,
        adj=torch.as_tensor(graph/(graph.sum(1,keepdims=True)+1e-6),dtype=torch.float32),
        node_ids=node_ids,active_nodes=active,metadata=metadata,preprocessing=metadata,
        manifest_sha256=manifest_digest,plan_sha256=plan_digest,train_allowed_stop=stop,
        train_raw_data=train_raw,train_raw_valid=source_valid[:stop].copy(),year=year,dataset=dataset)
