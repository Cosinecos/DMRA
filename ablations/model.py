"""Component removals and fixed-statistics controls for DMRA."""
import torch
from torch import nn
from torch.nn import functional as F

from dmra.model import DMRA, positive_number
from dmra.runtime.adapter_lora import positive_rank


ABLATIONS = (
    "full", "no_node_bank", "no_pattern_bank", "no_season", "no_lora",
    "no_dynamic_graph", "reuse_history_stats",
)
BASE_METHOD = "DMRA"
VARIANT = "dmra_ablation"


def canonical_ablation(value):
    if not isinstance(value, str):
        raise ValueError("ablation must be one of " + repr(ABLATIONS))
    value = value.strip().lower().replace("-", "_")
    if value not in ABLATIONS:
        raise ValueError("Unknown DMRA ablation: " + repr(value))
    return value


def ablation_description(value):
    value = canonical_ablation(value)
    return {
        "full": "Original DMRA, new week embeddings and GCN Adapter annual LoRA",
        "reuse_history_stats": "Remove all learned future-statistics prediction; per-window per-node historical residual mean/std normalize and restore the forecast",
        "no_node_bank": "No learnable node embedding or node-conditioned injection; the embedding-derived dynamic graph is also removed",
        "no_pattern_bank": "Remove the complete historical/future pattern branch; residual is X minus historical season",
        "no_season": "Remove both seasonal outputs, all TOD/DOW and new week embeddings, and all seasonal decoder parameters",
        "no_lora": "No annual LoRA A/B/bias in either MLP or graph Adapter; base Adapter retained and later frozen",
        "no_dynamic_graph": "Physical graph propagation only; node embeddings remain in all other paths",
    }[value]


class _ZeroPattern(nn.Module):
    """No parameters, projections, retrieval bank, or annual adapter slots."""

    def __init__(self, output_len):
        super().__init__()
        self.output_len = int(output_len)

    def forward(self, x, node_emb):
        return torch.zeros_like(x), x.new_zeros(*x.shape[:-1], self.output_len)


class _NodeFreeScaleLinear(nn.Linear):
    """Drop the constant-zero node columns from the existing scale input.

    The common forward constructs [H_G, E, mu_h, sigma_h]. In NO_NODE_BANK,
    E is always exactly zero, so its columns are unidentifiable dead weights.
    Keep only [H_G, mu_h, sigma_h] and copy all surviving weights verbatim.
    """

    @classmethod
    def from_existing(cls, previous, hidden):
        with torch.random.fork_rng(devices=[]):
            result = cls(hidden + 2, previous.out_features, previous.bias is not None)
        result = result.to(device=previous.weight.device, dtype=previous.weight.dtype)
        result.hidden = int(hidden)
        with torch.no_grad():
            result.weight.copy_(torch.cat((previous.weight[:, :hidden], previous.weight[:, -2:]), dim=1))
            if previous.bias is not None:
                result.bias.copy_(previous.bias)
        return result

    def forward(self, value):
        if value.shape[-1] != self.hidden * 2 + 2:
            raise ValueError("NO_NODE_BANK scale input must be [graph, zero-node, mean, std]")
        return F.linear(torch.cat((value[..., :self.hidden], value[..., -2:]), dim=-1),
                        self.weight, self.bias)


class DMRAAblation(DMRA):
    """One explicit ablation, with actual parameter removal and pinned lifecycle.

    First construct the original model so every retained parameter has the same
    initialization as the FULL control. Then prune disabled structures without
    consuming RNG. All normal flags keep their original meaning; the selected
    named ablation is enforced even if a contradictory source default is set.
    """

    def __init__(self, args):
        self.ablation = canonical_ablation(getattr(args, "ablation", "full"))
        super().__init__(args)
        if self.ablation == "reuse_history_stats":
            del self.scale_head
        if self.ablation == "no_pattern_bank":
            self.trend_branch = _ZeroPattern(self.output_len)
        if self.ablation == "no_season":
            for name in ("tod_embedding", "dow_embedding", "week_tod_embedding", "week_dow_embedding",
                         "week_interaction_proj", "season_node_proj", "season_time_proj",
                         "season_node_bias", "season_time_bias"):
                delattr(self, name)
        if self.ablation == "no_node_bank":
            original = self.node_embedding
            del self.node_embedding
            # A zero buffer preserves shape/device and stable-registry plumbing;
            # it is not a learnable node table and is never optimized.
            self.register_buffer("node_embedding", torch.zeros_like(original))
            # E=0 makes both low-rank node*time products identically zero. The
            # old node bias reduces to a shared scalar; retain only that scalar.
            self.season_node_constant = nn.Parameter(self.season_node_bias.bias.detach().clone())
            for name in ("season_node_proj", "season_time_proj", "season_node_bias"):
                delattr(self, name)
            self.scale_head[0] = _NodeFreeScaleLinear.from_existing(self.scale_head[0], self.hidden_dim)
        if self.ablation in ("no_node_bank", "no_dynamic_graph"):
            del self.node_graph_scale

    def _flag(self, name, default=False):
        if name == "use_residual_norm" and self.ablation == "reuse_history_stats":
            return True
        if name == "use_lora" and self.ablation == "no_lora":
            return False
        return super()._flag(name, default)

    def _ablation_enabled(self, name):
        if name == "ablate_v6_scale_head" and self.ablation == "reuse_history_stats":
            return True
        if name == "ablate_v6_season" and self.ablation == "no_season":
            return True
        if name == "ablate_v6_node_graph" and self.ablation in ("no_node_bank", "no_dynamic_graph"):
            return True
        return super()._ablation_enabled(name)

    def _seasonal_decode(self, node_context, tod_index, dow_index):
        if self.ablation == "no_season":
            raise RuntimeError("Seasonal decoder was removed")
        if self.ablation == "no_node_bank":
            time_term = self.season_time_bias(self._time_features(tod_index, dow_index)).squeeze(-1)
            return (time_term[:, None, :] + self.season_node_constant.reshape(1, 1, 1)).expand(
                -1, node_context.shape[0], -1)
        return super()._seasonal_decode(node_context, tod_index, dow_index)

    def _time_features(self, tod_index, dow_index):
        if self.ablation == "no_season":
            raise RuntimeError("All calendar embeddings were removed")
        return super()._time_features(tod_index, dow_index)

    def _dynamic_topk_graph(self, num_nodes, dtype):
        if self.ablation in ("no_node_bank", "no_dynamic_graph"):
            raise RuntimeError("Dynamic graph was removed in " + self.ablation)
        return super()._dynamic_topk_graph(num_nodes, dtype)

    def _node_context(self, num_nodes):
        if self.ablation == "no_node_bank":
            # Ignore even accidentally mutated buffers; there is no node input.
            return self.node_embedding.new_zeros(num_nodes, self.hidden_dim)
        return super()._node_context(num_nodes)

    def expand_adaptive_params(self, new_num_nodes, adj=None):
        new_num_nodes = int(new_num_nodes)
        if new_num_nodes < self.num_nodes:
            raise ValueError("DMRA ablations require append-only node identities")
        if new_num_nodes > self.num_nodes:
            old_nodes = self.num_nodes
            if self.ablation == "no_node_bank":
                self.node_embedding = self.node_embedding.new_zeros(new_num_nodes, self.hidden_dim)
            else:
                new_rows = self.node_embedding.new_empty(new_num_nodes - old_nodes, self.hidden_dim).uniform_(-.1, .1)
                if adj is not None:
                    graph = adj.to(device=self.node_embedding.device)
                    existing = self.node_embedding.detach()
                    for offset, node in enumerate(range(old_nodes, new_num_nodes)):
                        neighbor = graph[node, :old_nodes] > 0
                        if neighbor.any():
                            new_rows[offset] = existing[neighbor].mean(dim=0)
                self.node_embedding = nn.Parameter(torch.cat((self.node_embedding, new_rows), dim=0))
            self.num_nodes = new_num_nodes
            self._static_graph_cache = {}
        year, begin = int(self.args.year), int(self.args.begin_year)
        if self._flag("use_lora", True) and year > begin:
            years = range(begin + 1, year + 1) if self._flag("retrain_match_capacity", False) else (year,)
            for current in years:
                if current not in self._lora_years_added:
                    self.lora_mlp_branch.add_year(current)
                    self._lora_years_added.add(current)
                if self.ablation != "no_pattern_bank" and current not in self._trend_lora_added:
                    self.trend_branch.add_year(current)
                    self._trend_lora_added.add(current)
                if self._adapter_lora_enabled() and current not in self._adapter_lora_years_added:
                    self.gcn_adapter.add_year(current)
                    self._adapter_lora_years_added.add(current)

    def configure_incremental_training(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if isinstance(self.node_embedding, nn.Parameter):
            self.node_embedding.requires_grad_(True)
        if self.ablation != "no_season":
            for module in (self.tod_embedding, self.dow_embedding, self.week_tod_embedding,
                           self.week_dow_embedding, self.week_interaction_proj):
                module.requires_grad_(True)
        graph_scale = getattr(self, "node_graph_scale", None)
        if isinstance(graph_scale, nn.Parameter):
            graph_scale.requires_grad_(True)
        year = int(self.args.year)
        self.lora_mlp_branch.freeze_all_except_year(year)
        if self.ablation != "no_pattern_bank":
            self.trend_branch.freeze_all_except_year(year)
            self.trend_branch.bank.requires_grad_(True)
        self.gcn_adapter.freeze_all_except(year)
        self.branch_alpha_raw.requires_grad_(True)

    def restore_lora_slots(self, state):
        restored = super().restore_lora_slots(state)
        if self.ablation == "no_pattern_bank":
            self._trend_lora_added = set()
        return restored


class _MadridRegistryMixin:
    """Original stable registry and availability masking, with ablation dispatch."""

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
            self.registry_node_ids = torch.cat((self.registry_node_ids,
                                              self.registry_node_ids.new_full((new_num_nodes-old_nodes,), -1)))
            self.task_graph_active = torch.cat((self.task_graph_active,
                                              self.task_graph_active.new_ones(new_num_nodes-old_nodes)))

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
        if self._ablation_enabled("ablate_v6_node_graph"):
            raise RuntimeError("Dynamic graph was removed or disabled")
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


class MadridDMRAAblation(_MadridRegistryMixin, DMRAAblation):
    """Same six variants with Madrid stable IDs, active nodes and masking."""


def model_class(value="full", madrid=False):
    canonical_ablation(value)
    return MadridDMRAAblation if madrid else DMRAAblation


def method_for(value, madrid=False):
    value = canonical_ablation(value)
    return BASE_METHOD + "_ABLATION_" + value.upper() + ("_MADRID_MMD" if madrid else "")


def validate_embedding_config(config):
    canonical_ablation(config.get("ablation", "full"))
    for key, default in (("week_interaction_rank", 16), ("gcn_adapter_rank", 16),
                         ("gcn_adapter_lora_rank", 4), ("lora_rank", 4)):
        positive_rank(config.get(key, default), key)
    positive_number(config.get("bank_temperature", .3), "bank_temperature")
    positive_number(config.get("bank_qk_norm_eps", 1e-8), "bank_qk_norm_eps")
    for key in ("use_lora", "use_gcn_adapter_lora"):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(key + " must be a bool")


def embedding_architecture(args):
    value = canonical_ablation(getattr(args, "ablation", "full"))
    hidden = int(args.model["hidden_channel"])
    week = positive_rank(getattr(args, "week_interaction_rank", 16), "week_interaction_rank")
    bottleneck = positive_rank(getattr(args, "gcn_adapter_rank", 16), "gcn_adapter_rank")
    rank = positive_rank(getattr(args, "gcn_adapter_lora_rank", 4), "gcn_adapter_lora_rank")
    lora = value != "no_lora" and getattr(args, "use_lora", True)
    adapter_lora = lora and getattr(args, "use_gcn_adapter_lora", True)
    dynamic = value not in ("no_node_bank", "no_dynamic_graph") and not getattr(args, "ablate_v6_node_graph", False)
    season = value != "no_season" and not getattr(args, "ablate_v6_season", False)
    return {
        "embedding_variant": VARIANT, "embedding_design_version": 1,
        "ablation": value, "ablation_description": ablation_description(value),
        "source_design": BASE_METHOD,
        "hidden_dim": hidden,
        "learnable_node_bank": value != "no_node_bank",
        "node_bank_replacement": "parameter_table" if value != "no_node_bank" else "zero_buffer_no_node_injection",
        "pattern_branch": value != "no_pattern_bank",
        "bank_shared_key_value_shape": [int(getattr(args, "trend_bank_size", 64)), hidden]
            if value != "no_pattern_bank" else None,
        "bank_key_value_tied": value != "no_pattern_bank",
        "bank_retrieval": "cosine_shared_key_value_raw_value" if value != "no_pattern_bank" else "removed",
        "bank_temperature": positive_number(getattr(args, "bank_temperature", .3), "bank_temperature"),
        "bank_qk_norm_eps": positive_number(getattr(args, "bank_qk_norm_eps", 1e-8), "bank_qk_norm_eps"),
        "bank_temperature_trainable": False,
        "seasonal_branch": season,
        "calendar_embeddings_registered": value != "no_season",
        "week_interaction_rank": week if value != "no_season" else None,
        "week_interaction_parameters": week * (int(getattr(args, "steps_per_day", 288)) + 7 + hidden)
            if value != "no_season" else 0,
        "node_free_season": "shared_time_bias_plus_shared_scalar" if value == "no_node_bank" else None,
        "dynamic_graph": dynamic,
        "graph_propagation": "original_prior_plus_embedding_topk" if dynamic else "physical_only",
        "node_removal_dynamic_graph_coupling": value == "no_node_bank",
        "scale_head_input_dim": None if value == "reuse_history_stats" else (hidden + 2 if value == "no_node_bank" else hidden * 2 + 2),
        "scale_head_removed": value == "reuse_history_stats",
        "scale_head_present": value != "reuse_history_stats",
        "future_statistics": "per_sample_per_node_current_history_reuse" if value == "reuse_history_stats" else "original_future_statistics_head_or_explicit_fixed_stats_flag",
        "use_lora": bool(lora), "lora_rank": int(getattr(args, "lora_rank", 4)),
        "annual_mlp_branches": (["residual", "pattern"] if value != "no_pattern_bank" else ["residual"]) if lora else [],
        "gcn_adapter_rank": bottleneck, "gcn_adapter_base_parameters": 2 * hidden * bottleneck,
        "gcn_adapter_lora_rank": rank, "use_gcn_adapter_lora": bool(adapter_lora),
        "gcn_adapter_annual_parameters": 2 * rank * (hidden + bottleneck) if adapter_lora else 0,
        "lifecycle": "first_task_full_base_later_shared_parameters_and_current_annual_lora_only",
        "base_adapter_retained": True,
    }

