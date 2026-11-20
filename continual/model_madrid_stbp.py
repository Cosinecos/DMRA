"""Unchanged STBP layers with Madrid node identity/availability handling.

STBP does not consume physical edge weights. Its linear global attention and
global node/channel normalization must exclude training-inactive registry slots.
All-active inference uses exactly the original predictor's operations.
"""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from _upstream.stbp.model import STBP_Model


METHOD = "STBP_MADRID_2022_2026_MMD"


class MadridSTBP(STBP_Model):
    def __init__(self, args):
        super().__init__(args)
        self.register_buffer("registry_node_ids", torch.full((self.num_nodes,), -1, dtype=torch.long))
        self.register_buffer("task_graph_active", torch.ones(self.num_nodes, dtype=torch.bool))

    def expand_adaptive_params(self, new_num_nodes, adj=None):
        if new_num_nodes < self.num_nodes:
            raise ValueError("Madrid STBP requires append-only node registry positions")
        previous = self.num_nodes
        super().expand_adaptive_params(new_num_nodes)
        if new_num_nodes > previous:
            self.registry_node_ids = torch.cat((self.registry_node_ids, self.registry_node_ids.new_full((new_num_nodes-previous,), -1)))
            self.task_graph_active = torch.cat((self.task_graph_active, self.task_graph_active.new_ones(new_num_nodes-previous)))

    def set_task_nodes(self, node_ids, active_nodes):
        ids = torch.as_tensor(node_ids, device=self.pattern_bank.device)
        active = torch.as_tensor(active_nodes, device=self.pattern_bank.device)
        if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
            raise ValueError("Madrid node IDs must have integer dtype")
        if active.is_complex() or not ((active == 0) | (active == 1)).all():
            raise ValueError("Madrid availability must be boolean/0/1")
        ids, active = ids.long(), active.bool()
        if ids.shape != (self.num_nodes,) or active.shape != ids.shape:
            raise ValueError("Madrid node registry/availability shape mismatch")
        if (ids < 0).any() or ids.unique().numel() != self.num_nodes:
            raise ValueError("Madrid node IDs must be unique and nonnegative")
        bound = self.registry_node_ids >= 0
        if not torch.equal(ids[bound], self.registry_node_ids[bound]):
            raise ValueError("Madrid existing node registry changed")
        self.registry_node_ids.copy_(ids)
        self.task_graph_active.copy_(active)

    def configure_incremental_training(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name == "pattern_bank")

    def build_optimizer(self, lr):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=lr)

    def _predict_group(self, x, patterns):
        # Exact STBP_Model.forward layer order, without replacing parameters.
        batch, nodes, _ = x.shape
        value = F.relu(self.fconv1(x))
        memory = patterns.unsqueeze(0).expand(batch, *patterns.shape).permute(0, 2, 1).unsqueeze(-1)
        value = self.stmodule(value.permute(0, 2, 1).unsqueeze(-1), memory).squeeze(-1)
        value = self.fconv2(value.permute(0, 2, 1).squeeze(-1))
        value = value.reshape(-1, self.args.model["out_channel"]) + x.reshape(-1, self.args.model["in_channel"])
        value = self.fc(self.activation(value))
        return F.dropout(value, p=self.dropout, training=self.training).reshape(batch, nodes, self.args.y_len)

    def forward_dense(self, x):
        if x.ndim != 3 or x.shape[1] != self.num_nodes or x.shape[2] != self.args.model["in_channel"]:
            raise ValueError("Madrid STBP input does not match current registry/history")
        active = self.task_graph_active
        if bool(active.all()):
            return self._predict_group(x, self.pattern_bank)
        output = x.new_zeros(x.shape[0], self.num_nodes, self.args.y_len)
        active_ids = active.nonzero(as_tuple=True)[0]
        if active_ids.numel():
            output[:, active_ids] = self._predict_group(x[:, active_ids], self.pattern_bank[active_ids])
        dormant_ids = (~active).nonzero(as_tuple=True)[0]
        if dormant_ids.numel():
            # Each unavailable node is a separate N=1 group. Neither its values
            # nor its pattern vector can affect any other node's normalization
            # or global attention. Fallback predictions can still be evaluated
            # if an observation first appears in validation/test.
            batch, count = x.shape[0], dormant_ids.numel()
            local_x = x[:, dormant_ids].reshape(batch * count, 1, x.shape[-1])
            local_patterns = self.pattern_bank[dormant_ids]
            # _predict_group normally shares one node-pattern table across B;
            # the native N=1 computation below allows a different pattern per B.
            value = F.relu(self.fconv1(local_x))
            memory = local_patterns.unsqueeze(0).expand(batch, -1, -1).reshape(batch * count, -1)
            memory = memory.unsqueeze(-1).unsqueeze(-1)
            value = self.stmodule(value.permute(0, 2, 1).unsqueeze(-1), memory).squeeze(-1)
            value = self.fconv2(value.permute(0, 2, 1).squeeze(-1))
            value = value.reshape(-1, self.args.model["out_channel"]) + local_x.reshape(-1, x.shape[-1])
            value = F.dropout(self.fc(self.activation(value)), p=self.dropout, training=self.training)
            output[:, dormant_ids] = value.reshape(batch, count, self.args.y_len)
        return output

    def forward(self, data, adj):
        if adj.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("Madrid STBP graph and node registry differ")
        x = data.x.reshape(-1, self.num_nodes, self.args.model["in_channel"])
        return self.forward_dense(x).reshape(-1, self.args.y_len)


def registry_identity(state):
    bank, ids, active = (state.get(k) for k in ("pattern_bank", "registry_node_ids", "task_graph_active"))
    if not all(torch.is_tensor(v) for v in (bank, ids, active)):
        raise ValueError("STBP checkpoint missing pattern bank or Madrid registry")
    n = bank.shape[0]
    if ids.shape != (n,) or ids.dtype != torch.int64 or active.shape != (n,) or active.dtype != torch.bool:
        raise ValueError("STBP checkpoint Madrid registry shape/type mismatch")
    if (ids < 0).any() or ids.unique().numel() != n:
        raise ValueError("STBP checkpoint registry IDs are unbound or duplicated")
    return dict(nodes=n, node_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                graph_active_sha256=hashlib.sha256(active.cpu().numpy().tobytes()).hexdigest())
