"""Official TQNet adapted to the common forecasting interface."""
from types import SimpleNamespace

import torch
from torch import nn

from .vendor.model import Model


NATIVE_LOSS = "mse"
DEFAULTS = {
    "seq_len": 12,
    "pred_len": 12,
    "d_model": 512,
    "dropout": 0.1,
    "use_revin": True,
}
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class Forecaster(nn.Module):
    """Keep the native core and supply the *first forecast* daily cycle phase.

    The official data loader passes ``cycle_index[s_end]``, not the first
    history timestamp. The common loader already exposes calendar-only future
    indices, so use their first TOD when supplied. With contiguous regular
    history alone it is exactly ``(last_history_TOD + 1) % steps_per_day``.
    No future observations or labels enter this predictor.
    """

    def __init__(self, config, context):
        super().__init__()
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown TQNet model settings: {sorted(unknown)}")
        self.config = dict(DEFAULTS, **config)
        self.num_nodes = int(context["num_nodes"])
        self.steps_per_day = int(context["steps_per_day"])
        if self.num_nodes <= 0 or self.steps_per_day <= 0:
            raise ValueError("TQNet num_nodes and steps_per_day must be positive")
        for name in ("seq_len", "pred_len", "d_model"):
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"TQNet {name} must be a positive integer")
        if self.config["seq_len"] <= 1 or self.config["seq_len"] % 4:
            raise ValueError("TQNet history length must be >=4 and divisible by its four attention heads")
        value = self.config["dropout"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ValueError("TQNet dropout must be in [0,1)")
        if not isinstance(self.config["use_revin"], bool):
            raise ValueError("TQNet use_revin must be true/false")
        self.core = Model(SimpleNamespace(
            **self.config, enc_in=self.num_nodes,
            cycle=self.steps_per_day, model_type="mlp",
        ))

    def _calendar(self, value, x, length, name):
        if value is None or value.shape != (x.shape[0], length):
            raise ValueError(f"TQNet {name} must have shape [B,{length}]")
        if value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"TQNet {name} must have an integer dtype")
        if bool(((value < 0) | (value >= self.steps_per_day)).any()):
            raise ValueError(f"TQNet {name} must lie in [0,steps_per_day)")
        return value.to(device=x.device, dtype=torch.long)

    def _forecast_cycle(self, x, x_tod, y_tod):
        if y_tod is not None:
            future = self._calendar(y_tod, x, self.config["pred_len"], "future TOD")
            return future[:, 0]
        history = self._calendar(x_tod, x, self.config["seq_len"], "historical TOD")
        expected = (history[:, :1] + torch.arange(self.config["seq_len"], device=x.device)) % self.steps_per_day
        if not torch.equal(history, expected):
            raise ValueError("TQNet needs known future TOD for an irregular/noncontiguous historical clock")
        return (history[:, -1] + 1) % self.steps_per_day

    def forward(self, x, x_tod=None, x_dow=None, y_tod=None, y_dow=None):
        if x.ndim != 3 or x.shape[1:] != (self.num_nodes, self.config["seq_len"]):
            raise ValueError(f"TQNet expects [B,{self.num_nodes},{self.config['seq_len']}], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("TQNet values must have a floating-point dtype")
        cycle = self._forecast_cycle(x, x_tod, y_tod)
        prediction = self.core(x.transpose(1, 2), cycle)
        return prediction.transpose(1, 2).contiguous()


def build_model(config, context):
    return Forecaster(config, context)
