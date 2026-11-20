"""STBP on the independent 2022–2026 Madrid selected-month task chain."""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import random
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from madrid_runtime import main as core
from main_madrid_shift import DATASET, ROOT, check_data, task_records, selection_assets, sha256
from data_madrid_shift import load_madrid_shift_source, generate_madrid_shift_samples, load_cached_madrid_shift_inputs, ensure_node_prefix
from checkpoint_madrid_stbp import protocol
from model_madrid_stbp import METHOD
from trainer_madrid_stbp import run_task


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset',default=DATASET)
    result.add_argument('--conf',type=Path,default=Path(__file__).resolve().parent/'conf/madrid_stbp/MADRID_2022_2026_MMD.json')
    result.add_argument('--gpuid',type=int)
    result.add_argument('--seed',type=int)
    result.add_argument('--num-workers','--num_workers',dest='num_workers',type=int)
    result.add_argument('--epochs','--epoch',dest='epoch',type=int)
    result.add_argument('--batch-size','--batch_size',dest='batch_size',type=int)
    result.add_argument('--lr',type=float)
    result.add_argument('--normalize-x',type=core.boolean)
    result.add_argument('--normalization-scope',choices=('split','train'))
    result.add_argument('--init',type=core.boolean)
    result.add_argument('--train',type=core.boolean)
    result.add_argument('--data-process',dest='data_process',type=core.boolean)
    result.add_argument('--load-first-year',type=core.boolean)
    result.add_argument('--first-year-model-path')
    result.add_argument('--checkpoint-path')
    result.add_argument('--output-dir',type=Path)
    result.add_argument('--smoke',action='store_true')
    result.add_argument('--check-data',action='store_true')
    result.add_argument('--set',action='append',default=[],metavar='KEY=JSON')
    return result


def configuration(cli):
    core.ROOT=ROOT
    if cli.dataset.upper().replace('-','_') not in (DATASET,'MADRID','ALL'):
        raise ValueError('Madrid STBP only supports the selected2022-2026 dataset')
    cfg=core.configuration(cli,DATASET)
    cfg.update(dataset=DATASET,method=METHOD)
    if cfg['steps_per_day']!=96 or cfg['val_test_mix']:
        raise ValueError('Madrid STBP requires15-minute chronological data')
    if cfg['x_len']!=12 or cfg['y_len']!=12 or cfg['model']['in_channel']!=12 or cfg['model']['out_channel']!=12:
        raise ValueError('Madrid STBP requires the common12-to12 shape')
    if cfg['model']['hidden_channel']%2:
        raise ValueError('STBP hidden width must be even')
    if not np.isfinite(float(cfg['lr'])) or float(cfg['lr'])<=0:
        raise ValueError('Madrid STBP learning rate must be finite and positive')
    if not 2022<=cfg['begin_year']<=cfg['end_year']<=2026:
        raise ValueError('Madrid STBP task years must lie in2022-2026')
    if cfg.get('pred_dump'):
        raise ValueError('Legacy prediction dump does not carry Madrid observation masks')
    if not cfg['use_mae_loss'] or float(cfg['lambda_mae'])!=1 or any(cfg[k] for k in ('use_freq_loss','use_bank_ortho_loss','use_node_stability_loss','use_distillation_loss')) or any(float(cfg[k])!=0 for k in ('lambda_freq','lambda_bank_ortho','lambda_u','lambda_dist')):
        raise ValueError('Madrid STBP uses its native prediction MAE only; PSRAG auxiliary objectives are unsupported')
    if cfg['use_lora'] or cfg['use_residual_norm']:
        raise ValueError('Madrid STBP has no PSRAG LoRA or residual normalization branch')
    return cfg


def run(config):
    check_data(config)
    args=SimpleNamespace(**config)
    for field in ('raw_data_path','graph_path','save_data_path','model_path'):
        setattr(args,field,str(core.project_path(getattr(args,field))))
    manifest_path=Path(args.raw_data_path).parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    records=task_records(manifest);assets=selection_assets(manifest_path.parent,manifest)
    args.data_manifest_sha256=sha256(manifest_path)
    args.selection_result_sha256=assets['result_sha256']
    args.reference_anchor_sha256=assets['anchor_sha256']
    args.selected_months={year:record['month'] for year,record in records.items()}
    if args.gpuid>=0:
        if not torch.cuda.is_available() or args.gpuid>=torch.cuda.device_count():
            raise RuntimeError('Requested CUDA device unavailable; use --gpuid -1 for CPU')
        args.device=torch.device(f'cuda:{args.gpuid}')
    else:
        args.device=torch.device('cpu')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    args.path=str(Path(args.model_path)/f'{stamp}_seed{args.seed}_{uuid.uuid4().hex[:6]}')
    Path(args.path).mkdir(parents=True)
    args.logger=logging.getLogger(args.path);args.logger.setLevel(logging.INFO);args.logger.propagate=False
    for handler in (logging.StreamHandler(),logging.FileHandler(Path(args.path)/'train.log',encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));args.logger.addHandler(handler)
    cfg=dict(config,data_manifest_sha256=args.data_manifest_sha256,protocol=protocol(args))
    (Path(args.path)/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    args.logger.info('baseline=STBP device=%s run=%s settings=%s',args.device,args.path,cfg)
    results={};previous_ids=None;args.graph_size_list=[]
    try:
        for year in range(args.begin_year,args.end_year+1):
            args.year=year;args.task_month=records[year]['month'];args.task_label=records[year]['label']
            raw_path=Path(args.raw_data_path)/f'{year}.npz'
            source=load_madrid_shift_source(raw_path)
            args.source_sha256=sha256(raw_path);args.expected_node_ids=source['node_ids'].tolist()
            if previous_ids is not None:ensure_node_prefix(previous_ids,source['node_ids'])
            previous_ids=source['node_ids'].copy()
            with np.load(Path(args.graph_path)/f'{year}_adj.npz',allow_pickle=False) as archive:graph=archive['x']
            args.graph_size=len(source['node_ids']);args.graph_size_list.append(args.graph_size)
            args.expected_source_steps=len(source['x'])
            args.expected_source_first_timestamp=str(source['timestamps'][0]);args.expected_source_last_timestamp=str(source['timestamps'][-1])
            if year==args.begin_year:args.base_node_size=args.graph_size
            args.adj=torch.as_tensor(graph/(graph.sum(1,keepdims=True)+1e-6),device=args.device,dtype=torch.float32)
            scope=args.normalization_scope if args.normalize_x else 'raw'
            cache=Path(args.save_data_path)/f'explicit_mask_{scope}_L{args.x_len}_H{args.y_len}'/f'{year}.npz'
            if args.data_process or args.smoke:
                output=cache if args.cache_data and not args.smoke else None
                if output is not None:output.parent.mkdir(parents=True,exist_ok=True)
                inputs=generate_madrid_shift_samples(source['x'],source['valid'],source['timestamps'],graph=graph,node_ids=source['node_ids'],active_nodes=source['active_nodes'],coordinates=source.get('coordinates'),dst_invalid_clock=source['dst_invalid_clock'],train_rate=args.train_rate,val_rate=args.val_rate,test_rate=args.test_rate,x_len=args.x_len,y_len=args.y_len,normalize_x=args.normalize_x,normalization_scope=args.normalization_scope,sample_limits={'train':8,'val':4,'test':4} if args.smoke else None,savepath=output,source_sha256=args.source_sha256)
            else:
                inputs=load_cached_madrid_shift_inputs(cache,args)
            args.logger.info('task=%s nodes=%s training_active=%s samples=%s dst_excluded_windows=%s',args.task_label,args.graph_size,int(source['active_nodes'].sum()),{s:len(inputs[f'{s}_x']) for s in ('train','val','test')},json.loads(str(inputs['__metadata__']))['dst_excluded_window_counts'])
            results[str(year)]=dict(run_task(inputs,args),month=args.task_month,label=args.task_label)
            (Path(args.path)/'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
            del inputs,source
        average={}
        for horizon in results[str(args.begin_year)]['metrics']:
            average[horizon]={}
            for metric in ('MAE','RMSE','MAPE','WAPE'):
                values=[r['metrics'][horizon][metric] for r in results.values() if r['metrics'][horizon][metric] is not None]
                average[horizon][metric]=float(np.mean(values)) if values else None
        summary=dict(dataset=DATASET,baseline='STBP',run=args.path,smoke=args.smoke,tasks=list(results),average=average,protocol=protocol(args))
        (Path(args.path)/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        args.logger.info('COMPLETED %s',summary)
        return summary
    finally:
        for handler in list(args.logger.handlers):handler.close();args.logger.removeHandler(handler)


def main():
    cli=parser().parse_args();cfg=configuration(cli)
    if cli.check_data:
        print(json.dumps(dict(status='passed',baseline='STBP',dataset=DATASET,tasks=check_data(cfg)),indent=2));return
    print(json.dumps(dict(status='passed',runs=[run(cfg)]),indent=2))


if __name__=='__main__':main()
