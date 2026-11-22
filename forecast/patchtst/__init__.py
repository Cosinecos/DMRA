"""Official supervised PatchTST adapted to the common forecasting interface."""

# https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9
from types import SimpleNamespace

from torch import nn

from .vendor.model import Model

NATIVE_LOSS = "mse"
DEFAULTS = {
    "seq_len": 12, "pred_len": 12,
    "d_model": 128, "d_ff": 256, "e_layers": 3, "n_heads": 16,
    "dropout": 0.2, "fc_dropout": 0.2, "head_dropout": 0.0,
    "patch_len": 4, "stride": 2, "padding_patch": "end",
    "revin": True, "affine": False, "subtract_last": False,
    "individual": False, "decomposition": False, "kernel_size": 25,
}


class Forecaster(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown PatchTST model settings: {sorted(unknown)}")
        self.config = dict(DEFAULTS, **config)
        self.num_nodes = int(context["num_nodes"])
        for name in ("seq_len", "pred_len", "d_model", "d_ff", "e_layers", "n_heads", "patch_len", "stride", "kernel_size"):
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"PatchTST {name} must be a positive integer")
        if self.config["d_model"] % self.config["n_heads"]:
            raise ValueError("PatchTST d_model must be divisible by n_heads")
        if self.config["patch_len"] > self.config["seq_len"]:
            raise ValueError("PatchTST patch_len cannot exceed the supplied history length")
        if self.config["padding_patch"] not in (None, "end"):
            raise ValueError("PatchTST padding_patch must be None or 'end'")
        for name in ("revin", "affine", "subtract_last", "individual", "decomposition"):
            if not isinstance(self.config[name], bool):
                raise ValueError(f"PatchTST {name} must be a boolean")
        for name in ("dropout", "fc_dropout", "head_dropout"):
            if not 0 <= self.config[name] < 1:
                raise ValueError(f"PatchTST {name} must be in [0,1)")
        if self.config["decomposition"] and self.config["kernel_size"] % 2 == 0:
            raise ValueError("PatchTST decomposition kernel_size must be odd")
        self.core = Model(SimpleNamespace(**self.config, enc_in=self.num_nodes))

    def forward(self, x, x_tod=None, x_dow=None, y_tod=None, y_dow=None):
        if x.ndim != 3 or x.shape[1:] != (self.num_nodes, self.config["seq_len"]):
            raise ValueError(f"PatchTST expects [B,{self.num_nodes},{self.config['seq_len']}], got {tuple(x.shape)}")
        return self.core(x.transpose(1, 2)).transpose(1, 2).contiguous()


def build_model(config, context):
    return Forecaster(config, context)
