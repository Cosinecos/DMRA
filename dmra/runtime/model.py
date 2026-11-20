"""DMRA graph, residual, periodic, and continual adaptation layers.

Only PyTorch and the Python standard library are required. Parameter names match
its original implementation so saved model_state_dict checkpoints remain usable.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class PriorGraphGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, use_self=True):
        super().__init__()
        self.neighbor_proj = nn.Linear(input_dim, output_dim, bias=bias)
        if use_self:
            self.self_proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.self_proj = None

    def forward(self, x, adj):
        neighbor_feature = torch.matmul(adj.to(dtype=x.dtype), x)
        output = self.neighbor_proj(neighbor_feature)
        if self.self_proj is not None:
            output = output + self.self_proj(x)
        return output


class PriorGraphTCNBackbone(nn.Module):
    def __init__(self, input_len, hidden_dim, dropout, kernel_size=3, dilation=1):
        super().__init__()
        padding = int((kernel_size - 1) * dilation / 2)
        self.gcn1 = PriorGraphGCNLayer(input_len, hidden_dim, bias=True, use_self=True)
        self.tcn = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.gcn2 = PriorGraphGCNLayer(hidden_dim, hidden_dim, bias=True, use_self=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.shape
        x = F.relu(self.gcn1(x, adj))
        x = x.reshape(batch_size * num_nodes, 1, -1)
        x = self.tcn(x)
        x = x.reshape(batch_size, num_nodes, -1)
        x = self.gcn2(x, adj)
        x = self.dropout(F.gelu(x))
        return self.norm(x)


class FreTSFilter(nn.Module):
    """FreTS 风格频域 MLP (NeurIPS 2023)。

    与 FreqFilter (逐频率独立缩放, Y[k] = w[k]·X[k]) 的区别:
    频率间通过 Linear 全连接混合, Y = M · X, 让所有频率交互。

    简化版: 实部和虚部各用独立 Linear (无 bias), 不做实虚交叉。

    初始化: weight = I (单位矩阵)  →  训练第一步等价于不滤波。
    """

    def __init__(self, T):
        super().__init__()
        self.T = T
        n_freq = T // 2 + 1
        self.linear_real = nn.Linear(n_freq, n_freq, bias=False)
        self.linear_imag = nn.Linear(n_freq, n_freq, bias=False)
        with torch.no_grad():
            self.linear_real.weight.copy_(torch.eye(n_freq))
            self.linear_imag.weight.copy_(torch.eye(n_freq))

    def forward(self, x):
        # x: [..., T]
        x_f = torch.fft.rfft(x, dim=-1)             # [..., F] complex
        x_real = self.linear_real(x_f.real)         # [..., F]
        x_imag = self.linear_imag(x_f.imag)         # [..., F]
        x_f_out = torch.complex(x_real, x_imag)
        return torch.fft.irfft(x_f_out, n=self.T, dim=-1)


class CumulativeLoRALinear(nn.Module):
    """主干 W_0 + 跨年累积的 LoRA pair (A_t, B_t).

    forward:
        W_eff = W_0 + Σ_t (alpha/r) · A_t · B_t
        out   = W_eff · x + b_eff

    标准 LoRA 初始化: A 高斯小值, B 全 0  →  ΔW 起步 = 0, 不破坏老模型行为.

    跨年 add_year(t) 创建新一对 (A_t, B_t, db_t); 调用 freeze_all_except(t)
    只让 year_t 这对可训, 历史 W_0 + 历史 LoRA 全冻.
    """

    def __init__(self, in_dim, out_dim, bias=True, rank=4, alpha=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.lora_db = nn.ParameterDict()

    def add_year(self, year_id):
        key = str(year_id)
        if key in self.lora_A:
            return
        ref = self.linear.weight
        self.lora_A[key] = nn.Parameter(torch.randn(self.out_dim, self.rank, device=ref.device, dtype=ref.dtype) * 0.01)
        self.lora_B[key] = nn.Parameter(torch.zeros(self.rank, self.in_dim, device=ref.device, dtype=ref.dtype))
        self.lora_db[key] = nn.Parameter(torch.zeros(self.out_dim, device=ref.device, dtype=ref.dtype))

    def freeze_all_except(self, year_id):
        for p in self.linear.parameters():
            p.requires_grad = False
        target_key = str(year_id)
        for k in self.lora_A:
            req = (k == target_key)
            self.lora_A[k].requires_grad = req
            self.lora_B[k].requires_grad = req
            self.lora_db[k].requires_grad = req

    def forward(self, x):
        out = self.linear(x)
        if not self.lora_A:
            return out
        scale = self.alpha / self.rank
        for k in self.lora_A:
            mid = F.linear(x, self.lora_B[k])              # [..., rank]
            delta = F.linear(mid, self.lora_A[k]) * scale  # [..., out_dim]
            out = out + delta + self.lora_db[k]
        return out


class LoRAMLPResidualBranch(nn.Module):
    """LoRA-MLP 残差分支: 用 CumulativeLoRA 作为可累积层, 每层后注入节点嵌入加性偏置.

    流程:
        h_0 = up_proj(x_residual_norm)                          [B, N, T_in -> H]
        for ℓ = 1..L:
            h_ℓ = layer_ℓ(h_ℓ-1)                                [B, N, H]
            h_ℓ = h_ℓ + e_i                                     ★ 节点 emb 加性注入
            if ℓ != L: h_ℓ = GELU(h_ℓ)
        y_residual_branch = out_proj(h_L)                       [B, N, T_out]
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, rank=4, dropout=0.0):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        # 输入投影: T_in → H, 不挂 LoRA (输入维度是固定 T_in, 漂移由后续 LoRA 层承担)
        self.up_proj = CumulativeLoRALinear(in_dim, hidden_dim, bias=True, rank=rank)
        # 中间层: H → H, 全部挂 LoRA
        self.mid_layers = nn.ModuleList([
            CumulativeLoRALinear(hidden_dim, hidden_dim, bias=True, rank=rank)
            for _ in range(max(num_layers - 2, 0))
        ])
        # 输出投影: H → T_out
        self.out_proj = CumulativeLoRALinear(hidden_dim, out_dim, bias=True, rank=rank)
        self.dropout = nn.Dropout(dropout)

    def _all_lora_layers(self):
        layers = [self.up_proj]
        layers.extend(self.mid_layers)
        layers.append(self.out_proj)
        return layers

    def add_year(self, year_id):
        for layer in self._all_lora_layers():
            layer.add_year(year_id)

    def freeze_all_except_year(self, year_id):
        for layer in self._all_lora_layers():
            layer.freeze_all_except(year_id)

    def forward(self, x_res, node_emb):
        """
        x_res:    [B, N, T_in]    去季节归一化后的残差
        node_emb: [N, hidden_dim] 节点可学习嵌入 (与 hidden_dim 对齐)
        return:   [B, N, T_out]
        """
        # 节点 emb broadcast 到 batch 维: [N, H] → [1, N, H] 后让 PyTorch 广播
        e_b = node_emb.unsqueeze(0)

        # up_proj: T_in → H, 加节点 emb, GELU
        h = self.up_proj(x_res)                              # [B, N, H]
        h = h + e_b                                          # ★ 节点 emb 注入
        h = F.gelu(h)
        h = self.dropout(h)

        # 中间层: 每层后注入节点 emb 并过 GELU
        for layer in self.mid_layers:
            h = layer(h)
            h = h + e_b                                      # ★ 节点 emb 注入
            h = F.gelu(h)
            h = self.dropout(h)

        # out_proj: H → T_out, 不再加节点 emb (输出维度 = T_out, 跟 e_b 维度不匹配)
        return self.out_proj(h)


class FlowPatternTrendExtractor(nn.Module):
    """从原始输入提取趋势项 P (历史段) + 预测未来趋势项 P_future.

    流程 (历史段):
        h = up_proj(x)                          [B, N, H]    输入 → hidden
        h = h + e_i                             ★ 节点 emb 注入
        Q = q_proj(h)                           [B, N, H]    查询投影
        attn = softmax(Q · bank^T / √H)         [B, N, K]    K = bank size
        P_hist_h = attn · bank                  [B, N, H]    bank 同时作 Key 和 Value
        P_hist = hist_down(P_hist_h)            [B, N, T_in] 投影回时域

    流程 (未来段, 输入 = P_hist_h, 经多层 LoRA-MLP):
        h_0 = P_hist_h
        for ℓ = 1..L:
            h_ℓ = lora_layer_ℓ(h_ℓ-1)
            h_ℓ = h_ℓ + e_i                     ★ 每层后注入节点 emb
            if ℓ != L: h_ℓ = GELU(h_ℓ)
        P_future = out_proj(h_L)                [B, N, T_out]

    持续学习:
      - 流量模式 bank (K × H) 跨年持续可学习, 让模式库随数据演化
      - up_proj / q_proj / hist_down 跨年冻结 (结构性先验)
      - 未来段 LoRA-MLP 跨年累积 LoRA pair
    """

    def __init__(self, in_dim, hidden_dim, out_dim, bank_size=64,
                 future_layers=3, rank=4, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bank_size = bank_size

        # ── 历史段 ──
        # 输入投影: T_in → H
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        # Query 投影 (Q 来自 H + e_i)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 流量模式 bank: shape [bank_size, hidden_dim], 同时作为 attention 的 Key 和 Value
        self.bank = nn.Parameter(torch.randn(bank_size, hidden_dim) * 0.02)
        # bank 检索结果投影回时域: H → T_in
        self.hist_down = nn.Linear(hidden_dim, in_dim, bias=True)

        # ── 未来段: LoRA-MLP ──
        # 输入是 P_hist_h (维度 H), 经多层 LoRA-MLP, 出 T_out
        self.future_up = CumulativeLoRALinear(hidden_dim, hidden_dim, bias=True, rank=rank)
        self.future_mid = nn.ModuleList([
            CumulativeLoRALinear(hidden_dim, hidden_dim, bias=True, rank=rank)
            for _ in range(max(future_layers - 2, 0))
        ])
        self.future_out = CumulativeLoRALinear(hidden_dim, out_dim, bias=True, rank=rank)

        self.dropout = nn.Dropout(dropout)

    def _all_future_lora_layers(self):
        layers = [self.future_up]
        layers.extend(self.future_mid)
        layers.append(self.future_out)
        return layers

    def add_year(self, year_id):
        for layer in self._all_future_lora_layers():
            layer.add_year(year_id)

    def freeze_all_except_year(self, year_id):
        for layer in self._all_future_lora_layers():
            layer.freeze_all_except(year_id)

    def forward(self, x, node_emb):
        """
        x:        [B, N, T_in]    原始输入
        node_emb: [N, H]
        return: (P_hist [B,N,T_in], P_future [B,N,T_out])
        """
        e_b = node_emb.unsqueeze(0)                               # [1, N, H]

        # ── 历史段: 提取 P_hist ──
        h = self.up_proj(x)                                        # [B, N, H]
        h = h + e_b                                                # ★ 节点 emb 注入
        Q = self.q_proj(h)                                         # [B, N, H]

        # bank 检索: Q · bank^T → softmax → · bank (bank 同时作 K 和 V)
        scale = self.hidden_dim ** 0.5
        attn = torch.softmax(Q @ self.bank.T / scale, dim=-1)      # [B, N, K]
        P_hist_h = attn @ self.bank                                # [B, N, H]

        # 投影回时域得到趋势项
        P_hist = self.hist_down(P_hist_h)                          # [B, N, T_in]

        # ── 未来段: 用 P_hist_h 走 LoRA-MLP 出 T_out ──
        h = self.future_up(P_hist_h)                               # [B, N, H]
        h = h + e_b                                                # ★ 节点 emb 注入
        h = F.gelu(h)
        h = self.dropout(h)

        for layer in self.future_mid:
            h = layer(h)
            h = h + e_b                                            # ★ 每层后注入
            h = F.gelu(h)
            h = self.dropout(h)

        P_future = self.future_out(h)                              # [B, N, T_out]

        return P_hist, P_future


class ForecastBackbone(nn.Module):
    """Season + bank trend + two residual predictors, with cumulative task LoRA.

    ``use_residual_norm=False`` bypasses residual statistics, the scale head and
    scale restoration together. Inputs retain the scale supplied by the caller.
    The scale-head parameters stay registered for checkpoint compatibility.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.model["hidden_channel"]
        self.input_len = args.x_len
        self.output_len = args.y_len
        self.dropout = args.dropout
        self.time_emb_dim = self.hidden_dim
        self.steps_per_day = getattr(args, "steps_per_day", 288)
        self.season_rank = getattr(args, "season_rank", max(8, self.hidden_dim // 8))
        self.gate_rank = getattr(args, "gate_rank", max(8, self.hidden_dim // 8))
        self.res_norm_eps = getattr(args, "res_norm_eps", 1e-5)
        self.res_scale_clip = getattr(args, "res_scale_clip", 2.0)
        self.node_graph_topk = getattr(args, "node_graph_topk", getattr(args, "proto_graph_topk", 32))
        self.num_nodes = args.base_node_size

        self.node_embedding = nn.Parameter(
            torch.empty(self.num_nodes, self.hidden_dim).uniform_(-0.1, 0.1)
        )
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.hidden_dim)
        self.dow_embedding = nn.Embedding(7, self.hidden_dim)
        self.season_node_proj = nn.Linear(self.hidden_dim, self.season_rank, bias=False)
        self.season_time_proj = nn.Linear(self.hidden_dim, self.season_rank, bias=False)
        self.season_node_bias = nn.Linear(self.hidden_dim, 1)
        self.season_time_bias = nn.Linear(self.hidden_dim, 1)

        graph_scale = torch.tensor(float(getattr(args, "node_graph_scale", getattr(args, "proto_graph_scale", 1.0))))
        if self._flag("learnable_node_graph_scale", True):
            self.node_graph_scale = nn.Parameter(graph_scale)
        else:
            self.register_buffer("node_graph_scale", graph_scale)

        self.residual_head = nn.Linear(self.hidden_dim, self.output_len)
        self.scale_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)
        self.output_dropout = nn.Dropout(self.dropout)
        self.standard_gcn_backbone = PriorGraphTCNBackbone(
            self.input_len,
            self.hidden_dim,
            self.dropout,
            kernel_size=getattr(args, "standard_gcn_tcn_kernel_size", 3),
            dilation=getattr(args, "standard_gcn_tcn_dilation", 1),
        )
        self.freq_filter = FreTSFilter(T=self.input_len)

        rank = int(getattr(args, "lora_rank", 4))
        self.lora_mlp_branch = LoRAMLPResidualBranch(
            in_dim=self.input_len,
            hidden_dim=self.hidden_dim,
            out_dim=self.output_len,
            num_layers=int(getattr(args, "lora_mlp_layers", 3)),
            rank=rank,
            dropout=self.dropout,
        )
        self.branch_alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.trend_branch = FlowPatternTrendExtractor(
            in_dim=self.input_len,
            hidden_dim=self.hidden_dim,
            out_dim=self.output_len,
            bank_size=int(getattr(args, "trend_bank_size", 64)),
            future_layers=int(getattr(args, "trend_future_layers", 3)),
            rank=rank,
            dropout=self.dropout,
        )
        self._lora_years_added = set()
        self._trend_lora_added = set()
        self._static_graph_cache = {}
        self.reset_v6_profile()

    def _flag(self, name, default=False):
        value = getattr(self.args, name, default)
        if not isinstance(value, bool):
            raise TypeError("{} must be a bool, got {!r}. Parse configuration flags before constructing the model.".format(name, value))
        return value

    def _ablation_enabled(self, name):
        return self._flag(name)

    def count_parameters(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        self.args.logger.info("Total Parameters: %s", total)
        self.args.logger.info("Trainable Parameters: %s", trainable)

    def build_optimizer(self, base_lr, incremental=False):
        return optim.AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=base_lr,
        )

    def expand_adaptive_params(self, new_num_nodes, adj=None):
        if new_num_nodes > self.num_nodes:
            old_num_nodes = self.num_nodes
            new_rows = torch.empty(
                new_num_nodes - old_num_nodes,
                self.hidden_dim,
                dtype=self.node_embedding.dtype,
                device=self.node_embedding.device,
            ).uniform_(-0.1, 0.1)
            if adj is not None:
                adj = adj.to(device=self.node_embedding.device)
                old_embedding = self.node_embedding.detach()
                for offset, node in enumerate(range(old_num_nodes, new_num_nodes)):
                    neighbor_mask = adj[node, :old_num_nodes] > 0
                    if neighbor_mask.any():
                        new_rows[offset] = old_embedding[neighbor_mask].mean(dim=0)
            self.node_embedding = nn.Parameter(torch.cat([self.node_embedding, new_rows], dim=0))
            self.num_nodes = new_num_nodes
            self._static_graph_cache = {}

        # A fresh capacity-matched model recreates the entire adapter structure.
        # All its weights are newly initialized and remain trainable.
        year = int(self.args.year)
        if self._flag("use_lora", True) and year > int(self.args.begin_year):
            years = (range(int(self.args.begin_year) + 1, year + 1)
                     if self._flag("retrain_match_capacity", False) else (year,))
            for adapter_year in years:
                if adapter_year not in self._lora_years_added:
                    self.lora_mlp_branch.add_year(adapter_year)
                    self._lora_years_added.add(adapter_year)
                if adapter_year not in self._trend_lora_added:
                    self.trend_branch.add_year(adapter_year)
                    self._trend_lora_added.add(adapter_year)

    def configure_incremental_training(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.node_embedding.requires_grad = True
        for module in (self.tod_embedding, self.dow_embedding):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if isinstance(self.node_graph_scale, nn.Parameter):
            self.node_graph_scale.requires_grad = True
        year = int(self.args.year)
        self.lora_mlp_branch.freeze_all_except_year(year)
        self.trend_branch.freeze_all_except_year(year)
        self.branch_alpha_raw.requires_grad = True
        self.trend_branch.bank.requires_grad = True

    def _node_context(self, num_nodes):
        return self.node_embedding[:num_nodes]

    def _time_features(self, tod_index, dow_index):
        return self.tod_embedding(tod_index.long()) + self.dow_embedding(dow_index.long())

    def _seasonal_decode(self, node_context, tod_index, dow_index):
        time_feature = self._time_features(tod_index, dow_index)
        node_coeff = self.season_node_proj(node_context)
        time_coeff = self.season_time_proj(time_feature)
        seasonal = torch.einsum("nr,btr->bnt", node_coeff, time_coeff) / math.sqrt(self.season_rank)
        seasonal = seasonal + self.season_node_bias(node_context).squeeze(-1).unsqueeze(0).unsqueeze(-1)
        seasonal = seasonal + self.season_time_bias(time_feature).squeeze(-1).unsqueeze(1)
        return seasonal

    def _prior_graph(self, adj, dtype):
        graph = adj.to(dtype=dtype)
        degree = graph.sum(dim=-1, keepdim=True).clamp_min(self.res_norm_eps)
        return graph / degree

    def _dynamic_topk_graph(self, num_nodes, dtype):
        node_context = F.normalize(self._node_context(num_nodes), dim=-1)
        node_sim = (node_context @ node_context.transpose(0, 1)).to(dtype=dtype)
        topk = min(max(int(self.node_graph_topk), 1), num_nodes)
        if topk < num_nodes:
            topk_index = torch.topk(node_sim, k=topk, dim=-1).indices
            mask = torch.zeros_like(node_sim, dtype=torch.bool)
            mask.scatter_(dim=-1, index=topk_index, value=True)
            node_sim = node_sim.masked_fill(~mask, 0.0)
        return node_sim.clamp_min(0.0)

    def _fused_adj(self, adj, num_nodes, dtype):
        # In the original Dyn subclass this ablation flag was ignored.
        if self._ablation_enabled("ablate_v6_node_graph"):
            return self._prior_graph(adj, dtype)
        prior = adj.to(dtype=dtype)
        sim = self._dynamic_topk_graph(num_nodes, dtype)
        scale = self.node_graph_scale.clamp_min(0.0).to(dtype=dtype)
        fused = prior + scale * sim
        degree = fused.sum(dim=-1, keepdim=True).clamp_min(self.res_norm_eps)
        return fused / degree

    def forward(self, data, adj):
        num_nodes = adj.shape[0]
        x = data.x.reshape(-1, num_nodes, self.input_len)
        if not all(hasattr(data, name) for name in ("x_tod", "x_dow", "y_tod", "y_dow")):
            raise ValueError("Time indices x_tod/x_dow/y_tod/y_dow are required.")
        x_tod = data.x_tod.reshape(-1, self.input_len)
        x_dow = data.x_dow.reshape(-1, self.input_len)
        y_tod = data.y_tod.reshape(-1, self.output_len)
        y_dow = data.y_dow.reshape(-1, self.output_len)
        return self.forward_dense(x, x_tod, x_dow, y_tod, y_dow, adj).reshape(-1, self.output_len)

    def _adapt_gcn_feature(self, feature):
        """Identity extension point, before residual/scale heads."""
        return feature

    def forward_dense(self, x, x_tod, x_dow, y_tod, y_dow, adj):
        profile_total = self._profile_begin()
        if profile_total is not None:
            self._v6_profile_batches += 1
        num_nodes = adj.shape[0]
        node_context = self._node_context(num_nodes)
        if self._ablation_enabled("ablate_v6_time_features"):
            x_tod = torch.zeros_like(x_tod)
            x_dow = torch.zeros_like(x_dow)
            y_tod = torch.zeros_like(y_tod)
            y_dow = torch.zeros_like(y_dow)

        token = self._profile_begin()
        if self._ablation_enabled("ablate_v6_season"):
            x_season = torch.zeros_like(x)
        else:
            x_season = self._seasonal_decode(node_context, x_tod, x_dow)
        self._profile_end("season_x", token)
        token = self._profile_begin()
        if self._ablation_enabled("ablate_v6_season"):
            y_season = x.new_zeros(x.shape[0], num_nodes, self.output_len)
        else:
            y_season = self._seasonal_decode(node_context, y_tod, y_dow)
        self._profile_end("season_y", token)

        token = self._profile_begin()
        p_hist, p_future = self.trend_branch(x, node_context)
        self._profile_end("trend_branch", token)
        use_norm = self._flag("use_residual_norm", True)
        token = self._profile_begin()
        residual = x - x_season - p_hist
        if use_norm:
            res_mean = residual.mean(dim=-1, keepdim=True)
            res_std = residual.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.res_norm_eps)
            residual_input = (residual - res_mean) / res_std
        else:
            residual_input = residual
        self._profile_end("residual_norm" if use_norm else "residual", token)

        token = self._profile_begin()
        residual_input = self.freq_filter(residual_input)
        self._profile_end("freq_filter", token)
        backbone_token = self._profile_begin()
        token = self._profile_begin()
        fused_adj = self._fused_adj(adj, num_nodes, residual_input.dtype)
        self._profile_end("fused_adj", token)
        fused_feature = self.standard_gcn_backbone(residual_input, fused_adj)
        fused_feature = self._adapt_gcn_feature(fused_feature)
        self._profile_end("standard_gcn_backbone", backbone_token)
        token = self._profile_begin()
        z_a = self.residual_head(self.output_dropout(fused_feature))
        self._profile_end("residual_head", token)
        token = self._profile_begin()
        z_b = self.lora_mlp_branch(residual_input, node_context)
        self._profile_end("lora_mlp_branch", token)
        alpha = torch.sigmoid(self.branch_alpha_raw)
        residual_prediction = alpha * z_a + (1.0 - alpha) * z_b

        if use_norm:
            token = self._profile_begin()
            node_context_expanded = node_context.unsqueeze(0).expand(x.shape[0], -1, -1)
            scale_input = torch.cat([fused_feature, node_context_expanded, res_mean, res_std], dim=-1)
            if self._ablation_enabled("ablate_v6_scale_head"):
                future_res_mean = res_mean
                future_res_std = res_std
            else:
                scale_stat = self.scale_head(scale_input)
                future_res_mean = scale_stat[..., :1]
                future_res_std = F.softplus(scale_stat[..., 1:]) + self.res_norm_eps
            self._profile_end("scale_head", token)
            token = self._profile_begin()
            y_residual = residual_prediction * future_res_std + future_res_mean
            self._profile_end("restore", token)
        else:
            y_residual = residual_prediction

        token = self._profile_begin()
        prediction = y_season + y_residual + p_future
        self._profile_end("compose", token)
        self._profile_end("forward_total", profile_total)
        self._profile_flush()
        return prediction

    def _profile_enabled(self):
        return self._ablation_enabled("profile_v6_parts") and self._ablation_enabled("profile_v6_active")

    def _profile_uses_cuda(self):
        return str(getattr(self.args, "device", "")).startswith("cuda") and torch.cuda.is_available()

    def reset_v6_profile(self):
        self._v6_profile_stats = {}
        self._v6_profile_batches = 0
        self._v6_profile_pending = []

    def _profile_begin(self):
        if not self._profile_enabled():
            return None
        if self._profile_uses_cuda():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return ("cuda", start, end)
        return ("cpu", time.perf_counter())

    def _profile_end(self, name, token):
        if token is None:
            return
        if token[0] == "cuda":
            _, start, end = token
            end.record()
            self._v6_profile_pending.append((name, start, end))
        else:
            _, start = token
            self._v6_profile_stats[name] = self._v6_profile_stats.get(name, 0.0) + (time.perf_counter() - start)

    def _profile_flush(self):
        if not self._v6_profile_pending:
            return
        torch.cuda.synchronize()
        for name, start, end in self._v6_profile_pending:
            elapsed = start.elapsed_time(end) / 1000.0
            self._v6_profile_stats[name] = self._v6_profile_stats.get(name, 0.0) + elapsed
        self._v6_profile_pending = []

    def log_v6_profile(self, logger, epoch=None):
        self._profile_flush()
        if not self._v6_profile_stats:
            logger.info("[v6_profile] epoch=%s no data collected", epoch)
            return
        batches = max(self._v6_profile_batches, 1)
        logger.info("[v6_profile] epoch=%s batches=%s", epoch, batches)
        for name, total in sorted(self._v6_profile_stats.items()):
            logger.info("[v6_profile] %-18s total=%.6fs avg=%.3fms", name, total, total * 1000.0 / batches)
