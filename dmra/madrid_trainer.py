"""Train DMRA with explicit Madrid observation masks and observed zeros."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dmra.madrid_checkpoint import load_best_model, save_checkpoint
from dmra.runtime.data_madrid_shift import MadridShiftSpatioTemporalDataset
from dmra.madrid_model import model_class
from dmra.runtime.trainer import _check_finite, _enabled, needs_teacher


def observed_mae(pred,target,valid):
    valid=valid.bool() & torch.isfinite(target)
    safe_target=torch.where(valid,target,pred.detach())
    return torch.where(valid,(pred-safe_target).abs(),torch.zeros_like(pred)).sum()/valid.sum().clamp_min(1)


def compute_loss(pred,target,valid,model,args,batch=None,teacher_model=None):
    parts,weights={},{}
    if _enabled(args,"use_mae_loss","lambda_mae",1.):
        parts["mae"]=observed_mae(pred,target,valid)
        weights["mae"]=float(args.lambda_mae)
    if _enabled(args,"use_freq_loss","lambda_freq"):
        keep=valid.bool().all(-1)
        if keep.any():
            p=torch.fft.rfft(pred[keep],dim=-1,norm="ortho").abs()
            y=torch.fft.rfft(target[keep],dim=-1,norm="ortho").abs()
            if args.freq_remove_dc and p.shape[-1]>1:
                p,y=p[...,1:],y[...,1:]
            parts["freq"]=(p-y).abs().mean()
        else:
            parts["freq"]=pred.sum()*0
        weights["freq"]=float(args.lambda_freq)
    if _enabled(args,"use_bank_ortho_loss","lambda_bank_ortho"):
        bank=torch.nn.functional.normalize(model.trend_branch.bank,dim=-1)
        k=len(bank)
        parts["bank_ortho"]=(bank@bank.T-torch.eye(k,device=bank.device)).square().sum()/k**2
        weights["bank_ortho"]=float(args.lambda_bank_ortho)
    if teacher_model is not None:
        old=teacher_model.num_nodes
        if _enabled(args,"use_node_stability_loss","lambda_u"):
            parts["node_stability"]=(model.node_embedding[:old]-teacher_model.node_embedding.detach()).square().mean()
            weights["node_stability"]=float(args.lambda_u)
        if _enabled(args,"use_distillation_loss","lambda_dist"):
            if batch is None:
                raise ValueError("Madrid distillation requires the current source batch")
            x=batch.x.reshape(-1,args.graph_size,args.x_len)[:,:old]
            times=[getattr(batch,k).reshape(-1,length) for k,length in (("x_tod",args.x_len),("x_dow",args.x_len),("y_tod",args.y_len),("y_dow",args.y_len))]
            previous=args.prev_adj[:old,:old]
            with torch.no_grad():
                reference=teacher_model.forward_dense(x,*times,previous)
            student=model.forward_dense(x,*times,previous)
            observed=batch.y_valid.reshape(-1,args.graph_size,args.y_len)[:,:old]
            available=(model.task_graph_active[:old]&teacher_model.task_graph_active).reshape(1,old,1)
            keep=observed&available
            parts["distillation"]=torch.where(keep,(student-reference).square(),0).sum()/keep.sum().clamp_min(1)
            weights["distillation"]=float(args.lambda_dist)
    if not parts:
        raise ValueError("No active Madrid training objective")
    return sum(weights[k]*v for k,v in parts.items()),parts


def _loader(inputs,split,args,shuffle=False):
    ds=MadridShiftSpatioTemporalDataset(inputs,split,steps_per_day=args.steps_per_day)
    if not len(ds):
        raise ValueError(f"Empty Madrid {split} windows")
    return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.num_workers,pin_memory=args.device.type=="cuda")


def _empty_sums(length,device):
    # Count, absolute error, square error, label magnitude, APE, positive count,
    # observed zero count. Missing placeholders never enter any sum.
    return torch.zeros(7,length,device=device,dtype=torch.float64)


def _accumulate(sums,pred,target,valid):
    pred,target=pred.double(),target.double()
    valid=valid.bool()&torch.isfinite(target)
    positive=valid&(target>0)
    error=pred-target
    sums[0]+=valid.sum(0)
    sums[1]+=torch.where(valid,error.abs(),0).sum(0)
    sums[2]+=torch.where(valid,error.square(),0).sum(0)
    sums[3]+=torch.where(valid,target.abs(),0).sum(0)
    sums[4]+=torch.where(positive,error.abs()/target.clamp_min(1e-12),0).sum(0)
    sums[5]+=positive.sum(0)
    sums[6]+=(valid&(target==0)).sum(0)


def metrics_from_sums(sums):
    a=np.asarray(sums.cpu() if torch.is_tensor(sums) else sums,dtype=np.float64).cumsum(axis=1)
    prefixes=[]
    result={}
    for h in range(a.shape[1]):
        if not a[0,h]:
            raise ValueError("No observed validation/test labels")
        values=dict(MAE=float(a[1,h]/a[0,h]),RMSE=float(np.sqrt(a[2,h]/a[0,h])),
                    MAPE=float(100*a[4,h]/a[5,h]) if a[5,h] else None,
                    WAPE=float(100*a[1,h]/a[3,h]) if a[3,h] else None,
                    observed_targets=int(a[0,h]),zero_targets=int(a[6,h]),MAPE_targets=int(a[5,h]))
        prefixes.append(values)
        if h+1 in (3,6,a.shape[1]):
            result[str(h+1)]=values
    result["Avg"]={k:float(np.mean([r[k] for r in prefixes if r[k] is not None])) if any(r[k] is not None for r in prefixes) else None for k in ("MAE","RMSE","MAPE","WAPE")}
    result["Avg"].update({k:prefixes[-1][k] for k in ("observed_targets","zero_targets","MAPE_targets")})
    return result


def evaluate(model,args,loader,log=False):
    model.eval()
    sums=_empty_sums(args.y_len,args.device)
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(args.device)
            pred=model(batch,args.adj)
            _check_finite(pred,"Madrid evaluation prediction",args)
            _accumulate(sums,pred,batch.y,batch.y_valid)
    metrics=metrics_from_sums(sums)
    if log:
        for h,row in metrics.items():
            args.logger.info("test task=%s prefix=%s %s",args.year,h,row)
    return metrics


def run_task(inputs,args):
    task_dir=Path(args.path)/str(args.year)
    task_dir.mkdir(parents=True,exist_ok=True)
    loaders={s:_loader(inputs,s,args,shuffle=s=="train") for s in ("train","val","test")}
    teacher=None
    if not args.train or (args.load_first_year and args.year==args.begin_year):
        model,_=load_best_model(args)
        # Registry mismatches must fail, not silently evaluate a different cohort.
        model.set_task_nodes(inputs["node_ids"],inputs["active_nodes"])
        return dict(metrics=evaluate(model,args,loaders["test"],True),epochs=0)
    if args.year>args.begin_year:
        model,_=load_best_model(args)
        if needs_teacher(args):
            teacher=copy.deepcopy(model).eval()
            for p in teacher.parameters():
                p.requires_grad=False
        model.expand_adaptive_params(args.graph_size,args.adj)
        model.set_task_nodes(inputs["node_ids"],inputs["active_nodes"])
        model.configure_incremental_training()
    else:
        model=model_class(args)(args).to(args.device)
        model.expand_adaptive_params(args.graph_size,args.adj)
        model.set_task_nodes(inputs["node_ids"],inputs["active_nodes"])
    model.count_parameters()
    args.logger.info("calendar=actual zero_labels=valid graph_active=%s/%s lora_years=%s normalize_x=%s",
                     int(model.task_graph_active.sum()), model.num_nodes,
                     sorted(model._lora_years_added), args.normalize_x)
    optimizer=model.build_optimizer(args.lr,incremental=teacher is not None)
    best,best_path,counter=float("inf"),None,0
    started=time.perf_counter()
    for epoch in range(args.epoch):
        model.train()
        totals,parts_sum=[],{}
        for batch_index,batch in enumerate(loaders["train"]):
            batch=batch.to(args.device)
            label_free_objective=(_enabled(args,"use_bank_ortho_loss","lambda_bank_ortho") or
                                  (teacher is not None and _enabled(args,"use_node_stability_loss","lambda_u")))
            if not batch.y_valid.any() and not label_free_objective:
                # AdamW can move weights on a zero-gradient step; no observation
                # is not evidence to optimize a supervised model.
                continue
            optimizer.zero_grad(set_to_none=True)
            pred=model(batch,args.adj)
            _check_finite(pred,"Madrid training prediction",args,epoch,batch_index)
            loss,parts=compute_loss(pred,batch.y,batch.y_valid,model,args,batch,teacher)
            _check_finite(loss,"Madrid loss",args,epoch,batch_index)
            loss.backward()
            if args.smoke:
                for name,p in model.named_parameters():
                    if p.grad is not None:
                        _check_finite(p.grad,"Madrid gradient "+name,args,epoch,batch_index)
            if float(getattr(args,"grad_clip",0))>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(args.grad_clip))
            optimizer.step()
            totals.append(float(loss.detach()))
            for k,v in parts.items():
                parts_sum[k]=parts_sum.get(k,0)+float(v.detach())
        if not totals:
            raise RuntimeError("Madrid training has no observed labels or active label-free objective")
        val=evaluate(model,args,loaders["val"])[str(args.y_len)]["MAE"]
        args.logger.info("task=%s epoch=%s train=%.6f val_mae=%.6f loss_components=%s",args.year,epoch,np.mean(totals),val,{k:v/len(totals) for k,v in parts_sum.items()})
        if val<best:
            best,counter=val,0
            best_path=task_dir/f"{val:.8f}.pkl"
            save_checkpoint(model,args,best_path)
        else:
            counter+=1
            if counter>args.patience:
                break
    if best_path is None:
        raise RuntimeError("No finite Madrid validation checkpoint")
    # Exercise the complete strict restoration path, including protocol/registry.
    saved_path=getattr(args,"checkpoint_path","")
    args.checkpoint_path=str(best_path)
    model,_=load_best_model(args)
    args.checkpoint_path=saved_path
    result=dict(metrics=evaluate(model,args,loaders["test"],True),epochs=epoch+1,seconds=round(time.perf_counter()-started,3),best_val_mae=best,checkpoint=best_path.relative_to(Path(args.path)).as_posix(),checkpoint_selection="global_observed_validation_MAE",registered_nodes=model.num_nodes,training_graph_active_nodes=int(model.task_graph_active.sum()))
    return result
