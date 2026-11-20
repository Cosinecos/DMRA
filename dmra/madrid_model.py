"""DMRA with Madrid node registries and availability masks."""
import torch
from torch import nn
from torch.nn import functional as F

from dmra.model import DMRA


VARIANTS = ("dmra",)


def variant_name(args):
    value = getattr(args, "embedding_variant", None)
    if value not in VARIANTS:
        raise ValueError("embedding_variant must be dmra")
    return value


def method_name(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown independent embedding variant")
    return "DMRA"


class _MadridRegistryMixin:
    """Preserve stable IDs and isolate training-unavailable graph nodes."""
    def __init__(self, args):
        super().__init__(args)
        self.register_buffer("registry_node_ids", torch.full((self.num_nodes,), -1, dtype=torch.long))
        self.register_buffer("task_graph_active", torch.ones(self.num_nodes, dtype=torch.bool))

    def expand_adaptive_params(self, new_num_nodes, adj=None):
        old_nodes = self.num_nodes
        if new_num_nodes < old_nodes:
            raise ValueError("Madrid stable registry cannot shrink or reorder")
        super().expand_adaptive_params(new_num_nodes, adj)
        if new_num_nodes > old_nodes:
            self.registry_node_ids = torch.cat((self.registry_node_ids, self.registry_node_ids.new_full((new_num_nodes-old_nodes,), -1)))
            self.task_graph_active = torch.cat((self.task_graph_active, self.task_graph_active.new_ones(new_num_nodes-old_nodes)))

    def set_task_nodes(self, node_ids, active_nodes):
        ids = torch.as_tensor(node_ids, device=self.node_embedding.device)
        active = torch.as_tensor(active_nodes, device=self.node_embedding.device)
        if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
            raise ValueError("Node IDs must have an integer dtype")
        if active.is_complex() or not ((active == 0) | (active == 1)).all():
            raise ValueError("Graph availability must contain boolean/0/1 values")
        ids, active = ids.long(), active.bool()
        if ids.ndim != 1 or ids.numel() != self.num_nodes or active.shape != ids.shape:
            raise ValueError("Registry shape differs from expanded model")
        if ids.unique().numel() != ids.numel() or (ids < 0).any():
            raise ValueError("Node IDs must be unique and nonnegative")
        known = self.registry_node_ids >= 0
        if not torch.equal(ids[known], self.registry_node_ids[known]):
            raise ValueError("Existing node registry positions changed")
        self.registry_node_ids.copy_(ids)
        self.task_graph_active.copy_(active)

    def _masked_physical(self, adj, dtype):
        active = self.task_graph_active[:adj.shape[0]]
        graph = adj.to(dtype=dtype) * (active[:, None] & active[None, :]).to(dtype)
        return graph + torch.diag((~active).to(dtype))

    def _prior_graph(self, adj, dtype):
        graph = self._masked_physical(adj, dtype)
        return graph / graph.sum(-1, keepdim=True).clamp_min(self.res_norm_eps)

    def _fused_adj(self, adj, num_nodes, dtype):
        if self._ablation_enabled("ablate_v6_node_graph"):
            return self._prior_graph(adj, dtype)
        prior = self._masked_physical(adj, dtype)
        dynamic = self._dynamic_topk_graph(num_nodes, dtype)
        fused = prior + self.node_graph_scale.clamp_min(0).to(dtype) * dynamic
        return fused / fused.sum(-1, keepdim=True).clamp_min(self.res_norm_eps)

    def _dynamic_topk_graph(self, num_nodes, dtype):
        context = F.normalize(self._node_context(num_nodes), dim=-1)
        similarity = (context @ context.T).to(dtype)
        active = self.task_graph_active[:num_nodes]
        allowed = active[:, None] & active[None, :]
        scored = similarity.masked_fill(~allowed, float("-inf"))
        k = min(max(int(self.node_graph_topk), 1), num_nodes)
        if k < num_nodes:
            selected = torch.zeros_like(allowed).scatter_(1, scored.topk(k, dim=-1).indices, True)
            allowed = allowed & selected
        graph = torch.where(allowed, similarity.clamp_min(0), torch.zeros_like(similarity))
        return graph + torch.diag((~active).to(dtype))


class MadridDMRA(_MadridRegistryMixin, DMRA):
    """Annual LoRA inside GCN Adapter, with unchanged Madrid graph/registry rules."""


MODELS = {"dmra": MadridDMRA}


def model_class(args):
    return MODELS[variant_name(args)]
