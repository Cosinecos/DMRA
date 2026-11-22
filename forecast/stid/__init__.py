"""Official STID predictor on the common scalar-per-node forecasting interface."""

# https://github.com/GestaltCogTeam/STID/tree/e8b313bc591bdd0101a1619962c9b503e75127c0
import torch
from torch import nn

from .vendor.stid_arch import STID


NATIVE_LOSS = "mae"
DEFAULTS = {
    "seq_len": 12, "pred_len": 12, "input_dim": 1,
    "embed_dim": 32, "node_dim": 32,
    "temp_dim_tid": 32, "temp_dim_diw": 32, "num_layer": 3,
    "if_node": True, "if_T_i_D": True, "if_D_i_W": True,
}


class Forecaster(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown STID model settings: {sorted(unknown)}")
        self.config = dict(DEFAULTS, **config)
        self.num_nodes = int(context["num_nodes"])
        self.steps_per_day = int(context["steps_per_day"])
        if self.num_nodes <= 0 or self.steps_per_day <= 0:
            raise ValueError("STID num_nodes and steps_per_day must be positive")
        for name in ("seq_len", "pred_len", "input_dim", "embed_dim", "node_dim",
                     "temp_dim_tid", "temp_dim_diw", "num_layer"):
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"STID {name} must be a positive integer")
        if self.config["input_dim"] != 1:
            raise ValueError("The common scalar-per-node STID interface requires input_dim=1")
        for name in ("if_node", "if_T_i_D", "if_D_i_W"):
            if not isinstance(self.config[name], bool):
                raise ValueError(f"STID {name} must be a boolean")
        self.core = STID(
            num_nodes=self.num_nodes,
            input_len=self.config["seq_len"], output_len=self.config["pred_len"],
            time_of_day_size=self.steps_per_day, day_of_week_size=7,
            **{name: self.config[name] for name in (
                "input_dim", "embed_dim", "node_dim", "temp_dim_tid", "temp_dim_diw",
                "num_layer", "if_node", "if_T_i_D", "if_D_i_W")},
        )

    def _calendar_channel(self, indices, x, period, name, enabled):
        if not enabled:
            return torch.zeros_like(x.transpose(1, 2))
        if indices is None:
            raise ValueError(f"STID requires historical {name} indices when its embedding is enabled")
        if indices.shape != (x.shape[0], self.config["seq_len"]):
            raise ValueError(f"STID {name} indices must have shape [B,seq_len]")
        if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"STID {name} indices must have an integer dtype")
        if bool(((indices < 0) | (indices >= period)).any()):
            raise ValueError(f"STID {name} indices must lie in [0,{period})")
        # Official STID consumes fractional TOD and DOW channels, and only the
        # scalar value channel is flattened by its time-series projection.
        values = indices.to(device=x.device, dtype=x.dtype) / period
        return values.unsqueeze(-1).expand(-1, -1, self.num_nodes)

    def _history(self, x, x_tod, x_dow):
        if x.ndim != 3 or x.shape[1:] != (self.num_nodes, self.config["seq_len"]):
            raise ValueError(f"STID expects [B,{self.num_nodes},{self.config['seq_len']}], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("STID values must have a floating-point dtype")
        tod = self._calendar_channel(x_tod, x, self.steps_per_day, "TOD", self.config["if_T_i_D"])
        dow = self._calendar_channel(x_dow, x, 7, "DOW", self.config["if_D_i_W"])
        return torch.stack((x.transpose(1, 2), tod, dow), dim=-1)

    def forward(self, x, x_tod=None, x_dow=None, y_tod=None, y_dow=None):
        # Future calendar tensors are deliberately unused. No future values or
        # labels enter the official predictor; STID reads the last history ID.
        history = self._history(x, x_tod, x_dow)
        result = self.core(history, None, 0, 0, self.training)
        return result.squeeze(-1).transpose(1, 2).contiguous()


def build_model(config, context):
    return Forecaster(config, context)
