"""Dual Memories and Residual Adaptation for continual spatio-temporal forecasting."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from dmra.runtime.adapter_lora import GraphAdapterBackbone, positive_rank
from dmra.runtime.model import FlowPatternTrendExtractor


VARIANT = "dmra"
METHOD = "DMRA"


def positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


class PatternMemory(FlowPatternTrendExtractor):
    """One bank receives both the normalized-Key and unnormalized-Value gradients."""

    def install_retrieval(self, temperature, eps):
        self.bank_temperature = positive_number(temperature, "bank_temperature")
        self.bank_qk_norm_eps = positive_number(eps, "bank_qk_norm_eps")

    def retrieval_attention(self, query):
        query_unit = F.normalize(query, p=2, dim=-1, eps=self.bank_qk_norm_eps)
        key_unit = F.normalize(self.bank, p=2, dim=-1, eps=self.bank_qk_norm_eps)
        return torch.softmax(query_unit @ key_unit.T / self.bank_temperature, dim=-1)

    def forward(self, x, node_emb):
        e_b = node_emb.unsqueeze(0)
        query = self.q_proj(self.up_proj(x)+e_b)
        attention = self.retrieval_attention(query)
        pattern_hidden = attention @ self.bank
        pattern_history = self.hist_down(pattern_hidden)
        hidden = self.future_up(pattern_hidden)+e_b
        hidden = self.dropout(F.gelu(hidden))
        for layer in self.future_mid:
            hidden = self.dropout(F.gelu(layer(hidden)+e_b))
        pattern_future = self.future_out(hidden)
        return pattern_history, pattern_future


class DMRA(GraphAdapterBackbone):
    """Continual forecaster with shared-key/value pattern memory and calendar-conditioned periodic memory."""

    def __init__(self, args):
        rank = positive_rank(getattr(args, "week_interaction_rank", 16), "week_interaction_rank")
        temperature = positive_number(getattr(args, "bank_temperature", .3), "bank_temperature")
        eps = positive_number(getattr(args, "bank_qk_norm_eps", 1e-8), "bank_qk_norm_eps")
        super().__init__(args)
        self.trend_branch.__class__ = PatternMemory
        self.trend_branch.install_retrieval(temperature, eps)
        self.week_interaction_rank = rank
        # Match the independent new-embedding WEEK_INTERACTION initialization.
        # These are separate small tables, not products of the base TOD/DOW.
        with torch.random.fork_rng(devices=[]):
            self.week_tod_embedding = nn.Embedding(self.steps_per_day, rank)
            self.week_dow_embedding = nn.Embedding(7, rank)
            self.week_interaction_proj = nn.Linear(rank, self.hidden_dim, bias=False)
            nn.init.normal_(self.week_tod_embedding.weight, mean=0., std=.02)
            nn.init.normal_(self.week_dow_embedding.weight, mean=0., std=.02)
            nn.init.zeros_(self.week_interaction_proj.weight)

    def _time_features(self, tod_index, dow_index):
        additive = super()._time_features(tod_index, dow_index)
        joint = self.week_tod_embedding(tod_index.long()) * self.week_dow_embedding(dow_index.long())
        return additive + self.week_interaction_proj(joint)

    def configure_incremental_training(self):
        super().configure_incremental_training()
        for module in (self.week_tod_embedding, self.week_dow_embedding, self.week_interaction_proj):
            module.requires_grad_(True)


def model_class(variant):
    if variant != VARIANT:
        raise ValueError(f"Expected independent {VARIANT} variant, got {variant!r}")
    return DMRA


def method_for(variant):
    model_class(variant)
    return METHOD


def validate_embedding_config(config):
    model_class(config["embedding_variant"])
    for key, default in (("week_interaction_rank",16),("gcn_adapter_rank",16),("gcn_adapter_lora_rank",4)):
        positive_rank(config.get(key,default),key)
    positive_number(config.get("bank_temperature",.3),"bank_temperature")
    positive_number(config.get("bank_qk_norm_eps",1e-8),"bank_qk_norm_eps")
    if not isinstance(config.get("use_gcn_adapter_lora",True),bool):
        raise ValueError("use_gcn_adapter_lora must be a bool")


def embedding_architecture(args):
    model_class(args.embedding_variant)
    hidden = int(args.model["hidden_channel"])
    week = positive_rank(getattr(args,"week_interaction_rank",16),"week_interaction_rank")
    bottleneck = positive_rank(getattr(args,"gcn_adapter_rank",16),"gcn_adapter_rank")
    adapter_rank = positive_rank(getattr(args,"gcn_adapter_lora_rank",4),"gcn_adapter_lora_rank")
    flag = getattr(args,"use_gcn_adapter_lora",True)
    if not isinstance(flag,bool):
        raise ValueError("use_gcn_adapter_lora must be a bool")
    return dict(
        embedding_variant=VARIANT, embedding_design_version=1,
        composition="qknorm_new_week_embeddings_shared_key_value_gcn_adapter_annual_lora",
        bank_temperature=positive_number(getattr(args,"bank_temperature",.3),"bank_temperature"),
        bank_qk_norm_eps=positive_number(getattr(args,"bank_qk_norm_eps",1e-8),"bank_qk_norm_eps"),
        bank_temperature_trainable=False,
        bank_retrieval="L2_query_and_shared_bank_Key_only_Value_unnormalized",
        bank_shared_key_value_shape=[int(getattr(args,"trend_bank_size",64)),hidden],
        bank_key_value_tied=True, bank_key_is_parameter=False,
        bank_initialization="original_shared_bank_preserve_rng",
        bank_adaptation="same_shared_Key_Value_bank_trainable_every_task",
        bank_orthogonality_target="shared_bank_used_for_both_Key_and_Value",
        week_interaction_rank=week,
        week_interaction_design="base_TOD_plus_DOW_plus_W_new_u_TOD_times_new_v_DOW",
        week_interaction_position="seasonal_time_features_only",
        week_interaction_parameters=week*(int(getattr(args,"steps_per_day",288))+7+hidden),
        week_interaction_initialization="random_u_v_zero_output_projection_preserve_rng",
        week_interaction_adaptation="same_three_modules_trainable_every_task",
        gcn_adapter_rank=bottleneck,gcn_adapter_lora_rank=adapter_rank,use_gcn_adapter_lora=flag,
        gcn_adapter_hidden_dim=hidden,gcn_adapter_base_parameters=2*hidden*bottleneck,
        gcn_adapter_annual_parameters=2*adapter_rank*(hidden+bottleneck),
        gcn_adapter_position="after_backbone_LayerNorm_before_residual_and_scale_heads",
        gcn_adapter_lifecycle="first_task_train_base_later_freeze_base_train_current_annual_AB_only",
        gcn_adapter_bias="no_base_or_annual_bias",gcn_adapter_scaling="alpha_equals_rank",
    )
