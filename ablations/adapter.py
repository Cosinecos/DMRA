"""Continual training and strict best-state restore for single ablations."""
from copy import deepcopy

import torch
from torch.nn import functional as F

from .common import BaseAdapter, is_madrid, masked_loss
from .model import (DMRAAblation, MadridDMRAAblation,
                    embedding_architecture)


def enabled(args, flag, weight, default=0.):
    return bool(getattr(args,flag,False)) and float(getattr(args,weight,default))>0


class Adapter(BaseAdapter):
    def __init__(self,args):
        super().__init__(args);self.teacher=None

    def prepare_task(self,ctx):
        super().prepare_task(ctx);self.teacher=None
        if self.model is None:
            self.args.base_node_size=ctx.adj.shape[0]
            cls=MadridDMRAAblation if is_madrid(self.args) else DMRAAblation
            self.model=cls(self.args).to(self.args.device)
            # Strictly reconstructing a later checkpoint needs every earlier
            # annual slot before logging/restore. This affects structure only;
            # no optimizer or checkpoint inheritance is invoked here.
            later=int(self.args.year)>int(self.args.begin_year)
            original_match=self.args.retrain_match_capacity
            try:
                if later: self.args.retrain_match_capacity=True
                self.model.expand_adaptive_params(ctx.adj.shape[0],ctx.adj)
            finally:
                self.args.retrain_match_capacity=original_match
            if later: self.model.configure_incremental_training()
        else:
            if ctx.adj.shape[0]<self.model.num_nodes:
                raise ValueError('Continual node registry cannot shrink')
            if enabled(self.args,'use_node_stability_loss','lambda_u') or enabled(self.args,'use_distillation_loss','lambda_dist'):
                original=self.model.args
                try:
                    self.model.args=None
                    self.teacher=deepcopy(self.model).to(self.args.device).eval()
                finally:
                    self.model.args=original
                self.teacher.args=self.args;self.teacher.requires_grad_(False)
            self.model.args=self.args
            self.model.expand_adaptive_params(ctx.adj.shape[0],ctx.adj)
            self.model.configure_incremental_training()
        if hasattr(self.model,'set_task_nodes'):
            self.model.set_task_nodes(ctx.node_ids,ctx.active_nodes)
        self.model.args=self.args
        inventory={name:p.numel() for name,p in self.model.named_parameters() if p.requires_grad}
        self.args.logger.info('architecture=%s annual_lora_years=%s',embedding_architecture(self.args),self.model.saved_lora_years())
        self.args.logger.info('trainable_inventory=%s',inventory)

    def training_step(self,batch):
        args=self.args;prediction=self.predict(batch,training=True);parts={};weights={}
        if enabled(args,'use_mae_loss','lambda_mae',1.):
            parts['mae']=masked_loss(prediction,batch.y,batch.y_valid,args=args);weights['mae']=args.lambda_mae
        if enabled(args,'use_freq_loss','lambda_freq'):
            valid=batch.y_valid.bool() if is_madrid(args) else ~torch.isclose(batch.y,torch.zeros_like(batch.y))
            rows=valid.all(dim=-1)
            if rows.any():
                pred_fft=torch.fft.rfft(prediction[rows],dim=-1,norm='ortho').abs()
                true_fft=torch.fft.rfft(batch.y[rows],dim=-1,norm='ortho').abs()
                if args.freq_remove_dc: pred_fft,true_fft=pred_fft[...,1:],true_fft[...,1:]
                parts['freq']=(pred_fft-true_fft).abs().mean()
            else: parts['freq']=prediction.sum()*0
            weights['freq']=args.lambda_freq
        if enabled(args,'use_bank_ortho_loss','lambda_bank_ortho'):
            if args.ablation=='no_pattern_bank': raise ValueError('No bank for orthogonality loss')
            bank=F.normalize(self.model.trend_branch.bank,dim=-1);gram=bank@bank.T
            parts['bank_ortho']=(gram-torch.eye(len(bank),device=bank.device)).square().mean()
            weights['bank_ortho']=args.lambda_bank_ortho
        if self.teacher is not None:
            old=self.teacher.num_nodes
            if enabled(args,'use_node_stability_loss','lambda_u'):
                if args.ablation=='no_node_bank': raise ValueError('No learned node table for stability loss')
                delta=(self.model.node_embedding[:old]-self.teacher.node_embedding).square()
                if is_madrid(args):
                    active=self.teacher.task_graph_active[:old]&self.model.task_graph_active[:old];delta=delta[active]
                parts['node_stability']=delta.mean() if delta.numel() else prediction.sum()*0;weights['node_stability']=args.lambda_u
            if enabled(args,'use_distillation_loss','lambda_dist'):
                old_x=batch.x.reshape(-1,args.graph_size,args.x_len)[:,:old]
                calendars=[getattr(batch,name).reshape(-1,args.x_len if name.startswith('x') else args.y_len)
                    for name in ('x_tod','x_dow','y_tod','y_dow')]
                adj=args.prev_adj[:old,:old]
                with torch.no_grad(): reference=self.teacher.forward_dense(old_x,*calendars,adj)
                student=self.model.forward_dense(old_x,*calendars,adj);difference=(student-reference).square()
                if is_madrid(args):
                    active=self.teacher.task_graph_active[:old]&self.model.task_graph_active[:old]
                    valid=batch.y_valid.reshape(-1,args.graph_size,args.y_len)[:,:old]&active[None,:,None];difference=difference[valid]
                parts['distillation']=difference.mean() if difference.numel() else prediction.sum()*0;weights['distillation']=args.lambda_dist
        if not parts: raise ValueError('No active training loss')
        return sum(weights[k]*v for k,v in parts.items()),parts

    def extra_state(self):
        return {'annual_lora_years':self.model.saved_lora_years(),'architecture':embedding_architecture(self.args)}

    def restore(self,state):
        extra=state.get('adapter_state',{})
        if extra.get('architecture')!=embedding_architecture(self.args):
            raise ValueError('DMRA ablation architecture/checkpoint mismatch')
        self.model.restore_lora_slots(state['model_state_dict'])
        super().restore(state)

    def load_extra_state(self,state):
        if state.get('architecture')!=embedding_architecture(self.args): raise ValueError('Ablation architecture mismatch')
        if state.get('annual_lora_years')!=self.model.saved_lora_years(): raise ValueError('Annual slot years mismatch')
