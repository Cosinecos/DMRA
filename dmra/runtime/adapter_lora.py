"""Stage-specific bias-free low-rank updates to the graph feature adapter."""
import torch
from torch import nn
from torch.nn import functional as F

from dmra.runtime.model import (
    CumulativeLoRALinear,
    ForecastBackbone,
)




def positive_rank(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class AnnualAdapterLinear(nn.Linear):
    """Existing bias-free weight plus annual AB, with no annual bias vector."""

    def install_lora(self, rank):
        self.rank = positive_rank(rank, "gcn_adapter_lora_rank")
        self.alpha = self.rank
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()

    def add_year(self, year_id):
        key = str(int(year_id))
        if key not in self.lora_A:
            self.lora_A[key] = nn.Parameter(self.weight.new_empty(self.out_features, self.rank).normal_(std=.01))
            self.lora_B[key] = nn.Parameter(self.weight.new_zeros(self.rank, self.in_features))

    def freeze_all_except(self, year_id):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        key = str(int(year_id))
        if key in self.lora_A:
            self.lora_A[key].requires_grad_(True)
            self.lora_B[key].requires_grad_(True)

    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        for key in self.lora_A:
            output = output + F.linear(F.linear(x, self.lora_B[key]), self.lora_A[key]) * (self.alpha/self.rank)
        return output

    def merged_weight(self):
        """Return the exact effective linear weight; do not mutate or cache state."""
        result = self.weight
        for key in self.lora_A:
            result = result + (self.lora_A[key] @ self.lora_B[key]) * (self.alpha/self.rank)
        return result


class GCNAnnualLoRAAdapter(nn.Module):
    def __init__(self, hidden, bottleneck, lora_rank):
        super().__init__()
        # Match the previous GCN_ADAPTER's base weights and RNG stream exactly.
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(hidden, bottleneck, bias=False)
            self.up = nn.Linear(bottleneck, hidden, bias=False)
            nn.init.zeros_(self.up.weight)
            for layer in (self.down, self.up):
                layer.__class__ = AnnualAdapterLinear
                layer.install_lora(lora_rank)
        self.activation = nn.GELU()

    def forward(self, hidden):
        return hidden + self.up(self.activation(self.down(hidden)))

    def add_year(self, year):
        self.down.add_year(year)
        self.up.add_year(year)

    def freeze_all_except(self, year):
        self.down.freeze_all_except(year)
        self.up.freeze_all_except(year)


def expected_years(args):
    begin, saved = int(args.begin_year), int(args.year)
    if saved < begin:
        raise RuntimeError("Saved LoRA task precedes begin_year")
    return set(range(begin + 1, saved + 1))


class GraphAdapterBackbone(ForecastBackbone):
    def __init__(self, args):
        bottleneck = positive_rank(getattr(args, "gcn_adapter_rank", 16), "gcn_adapter_rank")
        rank = positive_rank(getattr(args, "gcn_adapter_lora_rank", 4), "gcn_adapter_lora_rank")
        super().__init__(args)
        self.gcn_adapter = GCNAnnualLoRAAdapter(self.hidden_dim, bottleneck, rank)
        self._adapter_lora_years_added = set()

    def _adapt_gcn_feature(self, hidden):
        return self.gcn_adapter(hidden)

    def expand_adaptive_params(self, new_num_nodes, adj=None):
        super().expand_adaptive_params(new_num_nodes, adj)
        year, begin = int(self.args.year), int(self.args.begin_year)
        if year > begin and year not in self._adapter_lora_years_added:
            self.gcn_adapter.add_year(year)
            self._adapter_lora_years_added.add(year)

    def configure_incremental_training(self):
        super().configure_incremental_training()
        # The Adapter's base D/U and old annual pairs remain frozen.
        self.gcn_adapter.freeze_all_except(int(self.args.year))

    def _lora_groups(self):
        mlp = [(name,module) for name,module in self.named_modules() if isinstance(module,CumulativeLoRALinear)]
        adapter = [("gcn_adapter.down", self.gcn_adapter.down), ("gcn_adapter.up",self.gcn_adapter.up)]
        return ((mlp, ("lora_A", "lora_B", "lora_db")), (adapter, ("lora_A", "lora_B")))

    def saved_lora_years(self):
        expected = expected_years(self.args)
        for layers, fields in self._lora_groups():
            for name,module in layers:
                for field in fields:
                    if set(getattr(module,field)) != {str(y) for y in expected}:
                        raise RuntimeError("Incomplete annual LoRA structure in "+name+"."+field)
        return sorted(expected)

    def restore_lora_slots(self, state):
        expected = expected_years(self.args)
        for layers, fields in self._lora_groups():
            for name,module in layers:
                for field in fields:
                    prefix = name+"."+field+"."
                    keys = {key[len(prefix):] for key in state if key.startswith(prefix)}
                    if keys != {str(y) for y in expected}:
                        raise RuntimeError("Incomplete/noncanonical annual LoRA years in "+name+"."+field)
                    for year in expected:
                        tensor = state[prefix+str(year)]
                        if isinstance(module,AnnualAdapterLinear):
                            shape = ((module.out_features,module.rank) if field=="lora_A"
                                     else (module.rank,module.in_features))
                        else:
                            shape = {"lora_A":(module.out_dim,module.rank),"lora_B":(module.rank,module.in_dim),
                                     "lora_db":(module.out_dim,)}[field]
                        if not torch.is_tensor(tensor) or tuple(tensor.shape)!=shape:
                            raise RuntimeError("Annual LoRA tensor shape differs in "+prefix+str(year))
                for year in sorted(expected):
                    module.add_year(year)
        self._lora_years_added = set(expected)
        self._trend_lora_added = set(expected)
        self._adapter_lora_years_added = set(expected)
        return sorted(expected)
