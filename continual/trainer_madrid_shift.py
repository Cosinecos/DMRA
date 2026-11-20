"""Shared utilities for the Madrid continual forecasting protocol."""

import numpy as np

import torch

from torch_geometric.loader import DataLoader

from data_madrid_shift import MadridShiftSpatioTemporalDataset

from madrid_runtime.trainer import _check_finite

def observed_mae(pred,target,valid):
    valid=valid.bool() & torch.isfinite(target)
    safe_target=torch.where(valid,target,pred.detach())
    return torch.where(valid,(pred-safe_target).abs(),torch.zeros_like(pred)).sum()/valid.sum().clamp_min(1)

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
