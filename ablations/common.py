"""DMRA ablation masks, best checkpoints and evaluation for full-data stages.

Only the comparison protocol is shared; model adapters retain native losses and
incremental algorithms. Madrid's explicit observation mask includes real zeros.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def is_madrid(args):
    return str(getattr(args, 'dataset', '')).upper().startswith('MADRID')


def masked_loss(pred, target, valid=None, kind='mae', args=None, reduction='mean'):
    """Retain native legacy zero handling; use explicit Madrid label validity."""
    kind = str(kind).lower()
    madrid = is_madrid(args)
    if not madrid and kind in {'mse', 'huber', 'smooth_l1'}:
        function = F.mse_loss if kind == 'mse' else F.smooth_l1_loss
        return function(pred, target, reduction=reduction)
    if madrid:
        if valid is None:
            raise ValueError('Madrid prediction losses require explicit observation validity')
        keep = valid.bool() & torch.isfinite(target)
    else:
        keep = ~torch.isclose(target, torch.zeros((), dtype=target.dtype, device=target.device), atol=5e-5, rtol=0)
        keep = keep & torch.isfinite(target)
    safe = torch.where(keep, target, pred.detach())
    if kind == 'mae':
        value = (pred-safe).abs()
    elif kind == 'mse':
        value = (pred-safe).square()
    elif kind in {'huber','smooth_l1'}:
        value = F.smooth_l1_loss(pred, safe, reduction='none')
    else:
        raise ValueError('Unknown prediction loss: '+kind)
    value = torch.where(keep, value, torch.zeros_like(value))
    if reduction == 'none': return value
    if reduction == 'sum': return value.sum()
    if reduction != 'mean': raise ValueError('Unknown reduction: '+str(reduction))
    return value.sum()/keep.sum().clamp_min(1)


class WindowDataset(Dataset):
    def __init__(self, inputs, split, steps_per_day=288):
        self.inputs, self.split, self.steps_per_day = inputs, split, int(steps_per_day)

    def __len__(self):
        return len(self.inputs[self.split+'_x'])

    def __getitem__(self, index):
        p, s = self.inputs, self.split
        x=np.asarray(p[s+'_x'][index], dtype=np.float32)
        y=np.asarray(p[s+'_y'][index], dtype=np.float32)
        data=Data(x=torch.from_numpy(x.T.copy()),y=torch.from_numpy(y.T.copy()))
        valid=p.get(s+'_y_valid')
        data.y_valid=torch.from_numpy((y!=0 if valid is None else np.asarray(valid[index],dtype=bool)).T.copy())
        history_valid=p.get(s+'_x_valid')
        data.x_valid=torch.from_numpy((np.isfinite(x) if history_valid is None else np.asarray(history_valid[index],dtype=bool)).T.copy())
        for phase,length in [('x',x.shape[0]),('y',y.shape[0])]:
            positions=p.get(s+'_'+phase+'_index')
            if s+'_'+phase+'_tod' in p:
                tod=np.asarray(p[s+'_'+phase+'_tod'][index],dtype=np.int64)
                dow=np.asarray(p[s+'_'+phase+'_dow'][index],dtype=np.int64)
            elif positions is not None:
                pos=np.asarray(positions[index],dtype=np.int64)
                tod,dow=pos%self.steps_per_day,(pos//self.steps_per_day)%7
            else:
                raise ValueError('Few-shot inputs must preserve absolute time indices')
            if tod.shape!=(length,) or dow.shape!=(length,):
                raise ValueError('Invalid calendar index shape')
            setattr(data,phase+'_tod',torch.from_numpy(tod.copy()))
            setattr(data,phase+'_dow',torch.from_numpy(dow.copy()))
        return data


def make_loader(inputs, split, args, shuffle=False):
    dataset=WindowDataset(inputs,split,args.steps_per_day)
    if not len(dataset): raise ValueError(f'Empty {split} split in task {args.year}')
    return DataLoader(dataset,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.num_workers,
                      pin_memory=args.device.type=='cuda')


def finite(value,name,args):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'{args.baseline}/{args.dataset}/{args.variant}/task={args.year}: non-finite {name}')


def cpu_state(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_state(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_state(v) for v in value)
    return value


def save_atomic(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(cpu_state(payload),temporary)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


class BaseAdapter:
    def __init__(self,args):
        self.args=args;self.model=None;self.context=None;self.train_inputs=None;self.train_adj=None
    def prepare_task(self,ctx):
        self.context=ctx;self.train_inputs=ctx.inputs;self.train_adj=ctx.adj
    def predict(self,batch,training=False):
        return self.model(batch,self.train_adj if training else self.context.adj)
    def training_step(self,batch):
        prediction=self.predict(batch,training=True)
        loss=masked_loss(prediction,batch.y,batch.y_valid,args=self.args)
        return loss,{'mae':loss}
    def valid_training_targets(self,batch):
        valid=batch.y_valid
        mapping=getattr(self,'mapping',None)
        if mapping is not None:
            valid=valid.reshape(-1,self.train_adj.shape[0],self.args.y_len)[:,mapping]
        return valid
    def build_optimizer(self):
        parameters=[p for p in self.model.parameters() if p.requires_grad]
        if not parameters: raise ValueError('No trainable parameters')
        return torch.optim.AdamW(parameters,lr=self.args.lr,weight_decay=getattr(self.args,'weight_decay',.01))
    def before_epoch(self,epoch): pass
    def after_task(self): pass
    def extra_state(self): return {}
    def load_extra_state(self,state): pass
    def snapshot(self):
        return {'model_state_dict':cpu_state(self.model.state_dict()),'adapter_state':cpu_state(self.extra_state())}
    def restore(self,state):
        self.model.load_state_dict(state['model_state_dict'],strict=True)
        self.load_extra_state(state.get('adapter_state',{}))


class PrefixMetrics:
    def __init__(self,horizon,madrid=False):
        self.horizon=int(horizon);self.madrid=bool(madrid)
        self.count=np.zeros(horizon,dtype=np.int64);self.absolute=np.zeros(horizon)
        self.squared=np.zeros(horizon);self.percentage=np.zeros(horizon)
        self.percentage_count=np.zeros(horizon,dtype=np.int64)
    def update(self,truth,prediction,valid=None):
        truth=np.asarray(truth,dtype=np.float64);prediction=np.asarray(prediction,dtype=np.float64)
        keep=(np.asarray(valid,dtype=bool)&np.isfinite(truth)) if self.madrid else ((truth!=0)&np.isfinite(truth))
        positive=keep&(truth>0) if self.madrid else keep
        delta=prediction-truth;axes=tuple(range(truth.ndim-1))
        self.count+=keep.sum(axis=axes)
        self.absolute+=np.where(keep,np.abs(delta),0).sum(axis=axes)
        self.squared+=np.where(keep,delta**2,0).sum(axis=axes)
        ratio=np.divide(np.abs(delta),np.abs(truth),out=np.zeros_like(truth),where=positive)
        self.percentage+=ratio.sum(axis=axes);self.percentage_count+=positive.sum(axis=axes)
    def result(self):
        count=np.cumsum(self.count)
        if np.any(count==0): raise ValueError('No observed evaluation targets')
        pc=np.cumsum(self.percentage_count)
        mae=np.cumsum(self.absolute)/count;rmse=np.sqrt(np.cumsum(self.squared)/count)
        mape=np.divide(100*np.cumsum(self.percentage),pc,out=np.full_like(mae,np.nan),where=pc>0)
        rows=[{'MAE':float(mae[i]),'RMSE':float(rmse[i]),'MAPE':float(mape[i]) if pc[i] else None} for i in range(self.horizon)]
        result={str(h):rows[h-1] for h in (3,6,self.horizon) if h<=self.horizon}
        result['Avg']={k:float(np.mean([v[k] for v in rows if v[k] is not None])) if any(v[k] is not None for v in rows) else None for k in rows[0]}
        return result


def validate(adapter,loader,args):
    adapter.model.eval();values=[];total=0.;count=0
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(args.device);prediction=adapter.predict(batch,training=False)
            if prediction.shape!=batch.y.shape: raise ValueError('Validation prediction/target shape mismatch')
            finite(prediction,'validation prediction',args)
            if is_madrid(args):
                valid=batch.y_valid.bool()&torch.isfinite(batch.y)
                total+=float(torch.where(valid,(prediction.double()-batch.y.double()).abs(),0).sum())
                count+=int(valid.sum())
            else:
                valid=batch.y!=0
                values.append(float(torch.where(valid,(prediction-batch.y).abs(),0).sum()/valid.sum().clamp_min(1)))
    if is_madrid(args):
        if not count: raise ValueError('No observed validation labels')
        return total/count
    return float(np.mean(values))


def evaluate(adapter,loader,args):
    adapter.model.eval();accumulator=PrefixMetrics(args.y_len,is_madrid(args))
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(args.device);pred=adapter.predict(batch,training=False)
            if pred.shape!=batch.y.shape: raise ValueError('Test prediction/target shape mismatch')
            finite(pred,'test prediction',args)
            accumulator.update(batch.y.cpu().numpy(),pred.cpu().numpy(),batch.y_valid.cpu().numpy())
    result=accumulator.result()
    for horizon,metrics in result.items(): args.logger.info('test task=%s prefix=%s %s',args.year,horizon,metrics)
    return result


def checkpoint_identity(args,ctx):
    from .model import embedding_architecture
    semantic={k:v for k,v in args.config_identity.items()
              if k not in {'output_root','output_dir','manifest_path','gpuid','num_workers','max_tasks'}}
    encoded=json.dumps(semantic,sort_keys=True,separators=(',',':')).encode()
    ids=np.asarray(ctx.node_ids,dtype=np.int64);active=np.asarray(ctx.active_nodes,dtype=bool)
    return dict(format_version=1,model=args.baseline,dataset=args.dataset,variant=args.variant,year=int(args.year),
                ablation=args.ablation,architecture=embedding_architecture(args),
                nodes=int(args.graph_size),config_sha256=hashlib.sha256(encoded).hexdigest(),
                manifest_sha256=str(ctx.manifest_sha256),plan_sha256=str(ctx.plan_sha256),
                node_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                active_nodes_sha256=hashlib.sha256(active.tobytes()).hexdigest(),
                graph_sha256=str(ctx.metadata['graph_sha256']),
                source_sha256=str(ctx.metadata['source_sha256']),
                normalize_x=bool(args.normalize_x),normalization_scope=args.normalization_scope,
                validation='global_observed_MAE' if is_madrid(args) else 'legacy_masked_MAE_batch_mean',
                target_scale='source',data_protocol=str(ctx.metadata['protocol']),
                training_fraction_of_entire_task=.6,
                train_allowed_stop=int(ctx.metadata['train_allowed_stop']))


def restore_checked(adapter,payload,args,ctx):
    expected=checkpoint_identity(args,ctx);metadata=payload.get('metadata',{})
    difference={k:(metadata.get(k),v) for k,v in expected.items() if metadata.get(k)!=v}
    if difference: raise ValueError('DMRA ablation checkpoint identity mismatch: '+str(difference))
    adapter.restore(payload)


def fit_task(adapter,ctx):
    args=ctx.args;started=time.perf_counter()
    BaseAdapter.prepare_task(adapter,ctx);adapter.prepare_task(ctx);adapter.model.to(args.device)
    training=make_loader(adapter.train_inputs,'train',args,True)
    validation=make_loader(ctx.inputs,'val',args);testing=make_loader(ctx.inputs,'test',args)
    optimizer=adapter.build_optimizer()
    total=sum(p.numel() for p in adapter.model.parameters());trainable=sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    args.logger.info('task=%s parameters=%s trainable=%s samples=%s plan_sha256=%s',args.year,total,trainable,
                     {s:len(ctx.inputs[s+'_x']) for s in ('train','val','test')},ctx.plan_sha256)
    best,best_epoch,counter=float('inf'),-1,0;history=[]
    checkpoint=Path(args.path)/str(args.year)/'best.pkl'
    for epoch in range(args.epoch):
        adapter.model.train();adapter.before_epoch(epoch);losses=[];components={}
        for batch in training:
            batch=batch.to(args.device)
            if is_madrid(args) and not adapter.valid_training_targets(batch).any(): continue
            optimizer.zero_grad(set_to_none=True);loss,parts=adapter.training_step(batch)
            finite(loss,'training loss',args);loss.backward()
            if args.smoke:
                for name,p in adapter.model.named_parameters():
                    if p.grad is not None: finite(p.grad,'gradient '+name,args)
            if getattr(args,'grad_clip',0)>0: torch.nn.utils.clip_grad_norm_(adapter.model.parameters(),args.grad_clip)
            optimizer.step();losses.append(float(loss.detach()))
            for name,value in parts.items(): components[name]=components.get(name,0)+float(value.detach() if torch.is_tensor(value) else value)
        if not losses: raise RuntimeError('No observed supervised training targets in this selected subgraph')
        score=validate(adapter,validation,args)
        if not np.isfinite(score): raise FloatingPointError('Non-finite validation MAE')
        parts={k:v/len(losses) for k,v in components.items()}
        row=dict(epoch=epoch,train=float(np.mean(losses)),val_mae=score,loss_components=parts);history.append(row)
        args.logger.info('task=%s epoch=%s train=%.6f val_mae=%.6f loss_components=%s',args.year,epoch,row['train'],score,parts)
        if score<best:
            best,best_epoch,counter=score,epoch,0;payload=adapter.snapshot()
            payload['metadata']={**checkpoint_identity(args,ctx),'best_val_mae':score,'epoch':epoch}
            save_atomic(payload,checkpoint)
        else:
            counter+=1
            if counter>args.patience: break
    if best_epoch<0: raise RuntimeError('No finite checkpoint saved')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    restore_checked(adapter,saved,args,ctx);reloaded=validate(adapter,validation,args)
    if not np.isclose(reloaded,best,atol=1e-5,rtol=1e-6): raise RuntimeError(f'Best checkpoint mismatch {best} vs {reloaded}')
    metrics=evaluate(adapter,testing,args);adapter.after_task()
    completed=adapter.snapshot();completed['metadata']=saved['metadata'];save_atomic(completed,checkpoint)
    return dict(metrics=metrics,epochs=epoch+1,best_epoch=best_epoch,best_val_mae=best,reloaded_val_mae=reloaded,
                checkpoint=checkpoint.relative_to(Path(args.path)).as_posix(),parameters=total,trainable_parameters=trainable,
                lr=args.lr,batch_size=args.batch_size,seconds=round(time.perf_counter()-started,3),history=history,
                manifest_sha256=ctx.manifest_sha256,plan_sha256=ctx.plan_sha256)
