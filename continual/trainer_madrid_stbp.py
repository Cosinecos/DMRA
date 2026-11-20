"""Native STBP annual pattern-only adaptation with Madrid observed-label MAE."""
from pathlib import Path
import time

import numpy as np
import torch

from checkpoint_madrid_stbp import load_checkpoint, save_checkpoint
from model_madrid_stbp import MadridSTBP
from trainer_madrid_shift import _loader, observed_mae, evaluate
from madrid_runtime.trainer import _check_finite


def run_task(inputs,args):
    task_dir = Path(args.path)/str(args.year); task_dir.mkdir(parents=True,exist_ok=True)
    loaders = {s:_loader(inputs,s,args,shuffle=s=='train') for s in ('train','val','test')}
    if not args.train:
        if not args.checkpoint_path:
            raise ValueError("Madrid STBP evaluation requires --checkpoint-path")
        model,_ = load_checkpoint(args.checkpoint_path,args,expected_year=args.year)
        model.set_task_nodes(inputs['node_ids'],inputs['active_nodes'])
        return dict(metrics=evaluate(model,args,loaders['test'],True),epochs=0)
    if args.load_first_year and args.year == args.begin_year:
        if not args.first_year_model_path:
            raise ValueError("--load-first-year requires --first-year-model-path")
        model,_ = load_checkpoint(args.first_year_model_path,args,expected_year=args.year)
        model.set_task_nodes(inputs['node_ids'],inputs['active_nodes'])
        args.stbp_previous_checkpoint = str(args.first_year_model_path)
        return dict(metrics=evaluate(model,args,loaders['test'],True),epochs=0,checkpoint=str(args.first_year_model_path))
    incremental = bool(args.init and args.year > args.begin_year)
    if incremental:
        previous = getattr(args,'stbp_previous_checkpoint',None)
        if not previous:
            raise ValueError("Madrid STBP incremental task is missing previous best checkpoint")
        model,_ = load_checkpoint(previous,args,expected_year=args.year-1)
        model.expand_adaptive_params(args.graph_size)
        model.set_task_nodes(inputs['node_ids'],inputs['active_nodes'])
        model.configure_incremental_training()
    else:
        args.base_node_size = args.graph_size
        model = MadridSTBP(args).to(args.device)
        model.set_task_nodes(inputs['node_ids'],inputs['active_nodes'])
    model.args = args
    model.count_parameters()
    args.logger.info('STBP task=%s calendar=actual zero_labels=valid pattern_only_incremental=%s training_active=%s/%s normalize_x=%s physical_edges=unused_by_native_STBP',
                     args.task_label,incremental,int(model.task_graph_active.sum()),model.num_nodes,args.normalize_x)
    optimizer = model.build_optimizer(args.lr)
    best,best_path,counter=float('inf'),None,0
    started=time.perf_counter()
    for epoch in range(args.epoch):
        model.train(); losses=[]
        for index,batch in enumerate(loaders['train']):
            batch=batch.to(args.device)
            if not batch.y_valid.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            pred=model(batch,args.adj)
            _check_finite(pred,'Madrid STBP prediction',args,epoch,index)
            loss=observed_mae(pred,batch.y,batch.y_valid)
            _check_finite(loss,'Madrid STBP MAE',args,epoch,index)
            loss.backward()
            if args.smoke:
                for name,p in model.named_parameters():
                    if p.grad is not None:
                        _check_finite(p.grad,'Madrid STBP gradient '+name,args,epoch,index)
            if float(getattr(args,'grad_clip',0))>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(args.grad_clip))
            optimizer.step(); losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError('Madrid STBP training has no observed labels')
        val=evaluate(model,args,loaders['val'])[str(args.y_len)]['MAE']
        args.logger.info('task=%s epoch=%s train=%.6f val_mae=%.6f loss_components=%s',args.year,epoch,np.mean(losses),val,dict(prediction=float(np.mean(losses))))
        if val < best:
            best,counter=val,0;best_path=task_dir/f'{val:.8f}.pkl'
            save_checkpoint(model,args,best_path,val)
        else:
            counter+=1
            if counter>args.patience:
                break
    if best_path is None:
        raise RuntimeError('Madrid STBP did not produce a finite validation checkpoint')
    model,_=load_checkpoint(best_path,args,expected_year=args.year)
    args.stbp_previous_checkpoint=str(best_path)
    return dict(metrics=evaluate(model,args,loaders['test'],True),epochs=epoch+1,
                seconds=round(time.perf_counter()-started,3),best_val_mae=best,
                checkpoint=best_path.relative_to(Path(args.path)).as_posix(),
                checkpoint_selection='global_observed_validation_MAE',registered_nodes=model.num_nodes,
                training_graph_active_nodes=int(model.task_graph_active.sum()),
                total_parameters=sum(p.numel() for p in model.parameters()),
                pattern_only_incremental=incremental)
