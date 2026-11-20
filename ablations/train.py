"""Train and evaluate full-data DMRA ablations."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from ablations.model import (
    ABLATIONS, ablation_description, embedding_architecture, method_for,
    validate_embedding_config,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('PEMS', 'AIR', 'CA_NODE_SHIFT', 'CA_MULTIYEAR_NODE_SHIFT',
            'MADRID_2022_2026_MMD')
DEFAULTS = {
    'embedding_variant': 'dmra_ablation', 'ablation': 'full',
    'bank_temperature': .3, 'bank_qk_norm_eps': 1e-8,
    'week_interaction_rank': 16, 'gcn_adapter_rank': 16,
    'gcn_adapter_lora_rank': 4, 'use_gcn_adapter_lora': True,
    'lora_rank': 4, 'lora_mlp_layers': 3, 'trend_future_layers': 3,
    'trend_bank_size': 64, 'season_rank': 24, 'gate_rank': 24,
    'seed': 42, 'gpuid': 0, 'num_workers': 0 if os.name == 'nt' else 4,
    'normalize_x': False, 'normalization_scope': 'split',
    'use_residual_norm': True, 'res_norm_eps': 1e-5, 'res_scale_clip': 2.,
    'use_mae_loss': True, 'use_freq_loss': False, 'use_bank_ortho_loss': False,
    'use_node_stability_loss': False, 'use_distillation_loss': False,
    'lambda_mae': 1., 'lambda_freq': 0., 'lambda_bank_ortho': 0.,
    'lambda_u': 0., 'lambda_dist': 0., 'freq_remove_dc': True,
    'ablate_v6_scale_head': False, 'ablate_v6_season': False,
    'ablate_v6_time_features': False, 'ablate_v6_node_graph': False,
    'node_graph_topk': 32, 'node_graph_scale': 1., 'learnable_node_graph_scale': True,
    'standard_gcn_tcn_kernel_size': 3, 'standard_gcn_tcn_dilation': 1,
    'steps_per_day': 288, 'pems_days': 31,
    'train_rate': .6, 'val_rate': .2, 'test_rate': .2, 'val_test_mix': False,
    'cache_data': False, 'data_process': True, 'train': True,
    'init': True, 'use_lora': True, 'retrain_match_capacity': False,
    'load_first_year': False, 'checkpoint_path': '', 'first_year_model_path': '',
    'epoch': 200, 'patience': 30, 'dropout': 0., 'weight_decay': .01,
    'grad_clip': 0., 'x_len': 12, 'y_len': 12, 'paral': 0,
    'model': {'in_channel': 12, 'out_channel': 12, 'hidden_channel': 192},
    'profile_v6_parts': False, 'profile_v6_epoch': -1, 'pred_dump': False,
    'lr_schedule': None, 'batch_size_schedule': None,
}


def boolean(value):
    if isinstance(value, bool): return value
    if str(value).lower() in {'true','yes','on','1'}: return True
    if str(value).lower() in {'false','no','off','0'}: return False
    raise argparse.ArgumentTypeError('expected true or false')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ablation', default='all', help='one ablation, comma-separated names, or all six ablations; full selects the reference model')
    p.add_argument('--dataset', default='all', help='dataset name/list or all; Madrid alias accepted')
    p.add_argument('--variant', choices=('raw','unified','both'), default='raw')
    p.add_argument('--protocol', choices=('full',), default='full')
    p.add_argument('--data-root', type=Path, default=Path(os.environ.get('DMRA_DATA_ROOT', ROOT/'data')), help='directory containing the five dataset folders')
    p.add_argument('--conf', type=Path, help='one dataset config; structural/protocol identity remains locked')
    p.add_argument('--gpuid', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=0 if os.name=='nt' else 4)
    p.add_argument('--epochs', '--epoch', dest='epoch', type=int)
    p.add_argument('--patience', type=int)
    p.add_argument('--batch-size', dest='batch_size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--normalization-scope', choices=('split','train'))
    p.add_argument('--normalize-x', type=boolean)
    p.add_argument('--bank-temperature', type=float)
    p.add_argument('--bank-qk-norm-eps', type=float)
    for field in ('week-interaction-rank','gcn-adapter-rank','gcn-adapter-lora-rank','lora-rank'):
        p.add_argument('--'+field, type=int)
    for field,dest in {
        'residual-norm':'use_residual_norm', 'fixed-stats':'ablate_v6_scale_head',
        'mae-loss':'use_mae_loss', 'freq-loss':'use_freq_loss',
        'bank-ortho-loss':'use_bank_ortho_loss', 'node-stability-loss':'use_node_stability_loss',
        'distillation-loss':'use_distillation_loss', 'lora':'use_lora',
        'gcn-adapter-lora':'use_gcn_adapter_lora', 'init':'init','train':'train',
        'load-first-year':'load_first_year', 'retrain-match-capacity':'retrain_match_capacity',
    }.items():
        p.add_argument('--'+field, dest=dest, type=boolean)
    p.add_argument('--output-dir', type=Path, default=Path('runs/ablations'))
    p.add_argument('--max-tasks', type=int, help='prefix stages only, for execution verification')
    p.add_argument('--smoke', action='store_true', help='real nodes, 4/2/2 windows, B2, at most two epochs')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--check-data', action='store_true')
    p.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
    return p


def canonical_dataset(value):
    name = str(value).upper().replace('-','_')
    name = {'CA_S': 'CA_NODE_SHIFT', 'CA_M': 'CA_MULTIYEAR_NODE_SHIFT',
            'MADRID': 'MADRID_2022_2026_MMD', 'MADRID_SHIFT': 'MADRID_2022_2026_MMD'}.get(name, name)
    if name not in DATASETS: raise ValueError('Unknown dataset: '+str(value))
    return name


def selected_datasets(cli):
    chosen = DATASETS if cli.dataset.lower() == 'all' else tuple(canonical_dataset(x) for x in cli.dataset.split(','))
    if len(set(chosen))!=len(chosen): raise ValueError('Duplicate dataset')
    if cli.conf and len(chosen)!=1: raise ValueError('--conf requires one dataset')
    return chosen


def selected_ablations(value):
    chosen = tuple(x for x in ABLATIONS if x!='full') if value.lower()=='all' else tuple(value.lower().split(','))
    if not chosen or any(x not in ABLATIONS for x in chosen):
        raise ValueError('Choose ablations: '+', '.join(ABLATIONS))
    if len(set(chosen))!=len(chosen): raise ValueError('Duplicate ablation')
    return chosen


def configuration(cli, ablation, dataset, variant):
    path = cli.conf or ROOT/'ablations'/'config'/f'{dataset}.json'
    if not path.is_absolute(): path=ROOT/path
    supplied=json.loads(path.read_text(encoding='utf-8-sig'))
    config=json.loads(json.dumps(DEFAULTS)); config.update(supplied)
    config['model']={**DEFAULTS['model'],**supplied.get('model',{})}
    for key,value in vars(cli).items():
        if key in DEFAULTS and key not in {'model','normalize_x'} and value is not None:
            config[key]=value
    if cli.normalize_x is not None and cli.normalize_x!=(variant=='unified'):
        raise ValueError('--normalize-x conflicts with --variant; select matching raw/unified')
    config.update(ablation=ablation,embedding_variant='dmra_ablation',
        dataset=dataset,variant=variant,protocol=cli.protocol,normalize_x=variant=='unified',
        output_root=str(cli.output_dir),
        max_tasks=cli.max_tasks,smoke=bool(cli.smoke),baseline='DMRA_'+ablation.upper(),
        method=method_for(ablation),target_scale='source')
    if cli.normalization_scope is None:
        config['normalization_scope']=supplied.get('normalization_scope','split')
    if ablation=='no_lora':
        if cli.use_lora is True or cli.use_gcn_adapter_lora is True:
            raise ValueError('no_lora conflicts with explicitly enabling annual LoRA')
        config.update(use_lora=False,use_gcn_adapter_lora=False)
    if ablation=='reuse_history_stats':
        if cli.use_residual_norm is False or cli.ablate_v6_scale_head is False:
            raise ValueError('reuse_history_stats requires --residual-norm true and --fixed-stats true')
        if not config['use_residual_norm']:
            raise ValueError('reuse_history_stats requires internal residual normalization and restoration')
        config.update(use_residual_norm=True,ablate_v6_scale_head=True)
    identity={k:config[k] for k in ('ablation','embedding_variant','dataset','variant','protocol',
        'normalize_x','baseline','method','target_scale','train_rate','val_rate','test_rate')}
    if ablation=='reuse_history_stats':
        identity.update(use_residual_norm=True,ablate_v6_scale_head=True)
    for setting in cli.set:
        if '=' not in setting: raise ValueError('--set needs KEY=JSON')
        key,text=setting.split('=',1)
        try: value=json.loads(text)
        except json.JSONDecodeError: value=text
        target=config;parts=key.split('.')
        for part in parts[:-1]:
            if not isinstance(target.get(part),dict): raise ValueError('Unknown setting '+key)
            target=target[part]
        if parts[-1] not in target: raise ValueError('Unknown setting '+key)
        target[parts[-1]]=value
    if any(config.get(k)!=v for k,v in identity.items()):
        raise ValueError('Ablation/dataset/protocol identity cannot be overridden')
    for key,value in DEFAULTS.items():
        if isinstance(value,bool): config[key]=boolean(config[key])
    if not config['init'] or not config['train'] or config['load_first_year'] or config['retrain_match_capacity']:
        raise ValueError('This entry trains a fresh continual chain: train/init true, load-first-year/retrain-match-capacity false')
    if config['x_len']!=12 or config['y_len']!=12:
        raise ValueError('Source protocols use 12-to-12 windows')
    for key in ('raw_data_path','graph_path','steps_per_day','pems_days','begin_year','end_year'):
        if key in supplied and key in {'raw_data_path','graph_path','steps_per_day','pems_days','begin_year'} and config[key]!=supplied[key]:
            raise ValueError('Use a separate data protocol for changing '+key)
    if config['end_year']>supplied['end_year'] or config['end_year']<config['begin_year']:
        raise ValueError('end_year must select a nonempty prefix of the configured stages')
    for key in ('epoch','batch_size'):
        if isinstance(config[key],bool) or int(config[key])<=0: raise ValueError(key+' must be positive')
    if config['patience']<0 or config['num_workers']<0: raise ValueError('patience/workers must be nonnegative')
    if not math.isfinite(float(config['lr'])) or config['lr']<=0: raise ValueError('lr must be finite and positive')
    for key in ('lambda_mae','lambda_freq','lambda_bank_ortho','lambda_u','lambda_dist'):
        if not math.isfinite(float(config[key])) or config[key]<0: raise ValueError(key+' must be finite and nonnegative')
    for flag in ('ablate_v6_season','ablate_v6_time_features','ablate_v6_node_graph'):
        if config[flag]: raise ValueError('Single ablation identity forbids extra '+flag)
    if config['normalization_scope'] not in {'split','train'}: raise ValueError('invalid normalization scope')
    if config['val_test_mix']: raise ValueError('Chronological validation/test splits are required')
    if ablation=='no_lora' and (config['use_lora'] or config['use_gcn_adapter_lora']):
        raise ValueError('no_lora cannot enable any annual LoRA')
    if ablation=='no_node_bank' and config['use_node_stability_loss'] and config['lambda_u']>0:
        raise ValueError('Node stability loss requires a learned node table, absent in no_node_bank')
    if ablation=='no_pattern_bank' and config['use_bank_ortho_loss'] and config['lambda_bank_ortho']>0:
        raise ValueError('Bank orthogonality requires a pattern bank, absent in no_pattern_bank')
    if cli.max_tasks is not None and cli.max_tasks<=0: raise ValueError('--max-tasks must be positive')
    config['seed']=cli.seed;config['gpuid']=cli.gpuid;config['num_workers']=cli.num_workers
    if cli.smoke: config.update(epoch=min(config['epoch'],2),batch_size=2,num_workers=0)
    for key in ('raw_data_path', 'graph_path'):
        relative = Path(config[key]).relative_to('data')
        config[key] = str(cli.data_root.resolve() / relative)
    validate_embedding_config(config)
    return config


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run(config):
    from ablations.adapter import Adapter
    from ablations.common import fit_task
    from ablations.data import load_context
    args=SimpleNamespace(**config)
    if args.gpuid<0: args.device=torch.device('cpu')
    else:
        if not torch.cuda.is_available() or args.gpuid>=torch.cuda.device_count():
            raise RuntimeError('Requested CUDA device is unavailable; use --gpuid -1 explicitly for CPU')
        args.device=torch.device(f'cuda:{args.gpuid}')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=Path(args.output_root)
    if not root.is_absolute(): root=ROOT/root
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    args.path=str(root/args.protocol/args.ablation/args.variant/args.dataset/f'{stamp}_seed{args.seed}_{uuid.uuid4().hex[:6]}')
    Path(args.path).mkdir(parents=True)
    args.logger=logging.getLogger(args.path);args.logger.setLevel(logging.INFO);args.logger.propagate=False
    for handler in (logging.StreamHandler(),logging.FileHandler(Path(args.path)/'train.log',encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));args.logger.addHandler(handler)
    args.config_identity=json.loads(json.dumps(config))
    write_json(Path(args.path)/'config.json',config)
    args.logger.info('ablation=%s protocol=%s dataset=%s device=%s run=%s settings=%s',
        args.ablation,args.protocol,args.dataset,args.device,args.path,config)
    args.logger.info('definition=%s',ablation_description(args.ablation))
    adapter=Adapter(args);results={};previous_adj=None
    years=list(range(args.begin_year,args.end_year+1))
    if args.max_tasks is not None: years=years[:args.max_tasks]
    try:
        for year in years:
            args.year=year
            ctx=load_context(ROOT,config,year,smoke=args.smoke)
            ctx.args=args;ctx.logger=args.logger;ctx.adj=ctx.adj.to(args.device)
            args.graph_size=ctx.adj.shape[0];args.adj=ctx.adj;args.sub_adj=ctx.adj;args.prev_adj=previous_adj
            if not results: args.base_node_size=args.graph_size
            args.logger.info('task=%s nodes=%s preprocessing=%s',year,args.graph_size,ctx.metadata)
            results[str(year)]=fit_task(adapter,ctx)
            write_json(Path(args.path)/'results.json',results)
            previous_adj=ctx.adj
        average={}
        for horizon in next(iter(results.values()))['metrics']:
            average[horizon]={}
            for metric in ('MAE','RMSE','MAPE'):
                values=[row['metrics'][horizon][metric] for row in results.values() if row['metrics'][horizon][metric] is not None]
                average[horizon][metric]=float(np.mean(values)) if values else None
        summary=dict(model=args.method,ablation=args.ablation,protocol=args.protocol,
            dataset=args.dataset,variant=args.variant,tasks=list(results),average=average,
            run=args.path,smoke=args.smoke,architecture=embedding_architecture(args))
        write_json(Path(args.path)/'summary.json',summary);args.logger.info('COMPLETED %s',summary)
        return summary
    finally:
        for handler in list(args.logger.handlers): handler.close();args.logger.removeHandler(handler)
        del adapter;gc.collect()
        if args.device.type=='cuda': torch.cuda.empty_cache()


def main():
    cli=parser().parse_args()
    configs=[configuration(cli,a,d,v) for a in selected_ablations(cli.ablation)
        for d in selected_datasets(cli) for v in (('raw','unified') if cli.variant=='both' else (cli.variant,))]
    if cli.plan:
        print(json.dumps(configs,ensure_ascii=False,indent=2));return
    from ablations.data import check_sources
    checked=set()
    for config in configs:
        key=(config['protocol'],config['dataset'])
        if key not in checked:
            missing=check_sources(ROOT,config)
            if missing: raise FileNotFoundError('Missing source files:\n'+'\n'.join(map(str,missing)))
            checked.add(key)
    if cli.check_data:
        print(json.dumps({'status':'passed','dataset_protocols':len(checked),'configurations':len(configs)}));return
    reports=[run(config) for config in configs]
    print(json.dumps({'status':'passed','runs':reports},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
