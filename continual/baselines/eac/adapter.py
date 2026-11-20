"""EAC's low-rank node prompts with its original incremental freezing rule."""

import torch.nn.functional as F

from baselines.common import BaseAdapter
from metrics import MAE_torch
from _upstream.eac.model import EAC_Model


DEFAULTS = {
    "dropout": 0.0,
    "lr": 0.03,
    "batch_size": 64,
    "epoch": 100,
    "patience": 5,
    "rank": 6,
    "loss": "mse",
    "model": {"in_channel": 12, "out_channel": 12, "hidden_channel": 64},
    "tcn": {"in_channel": 1, "out_channel": 1, "kernel_size": 3, "dilation": 1},
}


class Adapter(BaseAdapter):
    def prepare_task(self, ctx):
        super().prepare_task(ctx)
        # Retain the original model's argument names without creating another
        # hidden-width setting that can silently disagree with --set model.*.
        self.args.gcn = dict(self.args.model)
        if self.model is None or not getattr(self.args, "init", True):
            self.args.base_node_size = ctx.adj.shape[0]
            self.model = EAC_Model(self.args).to(self.args.device)
        else:
            if ctx.adj.shape[0] < self.model.num_nodes:
                raise ValueError("EAC requires append-only node identities between tasks")
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(not any(part in name for part in ("gcn1", "tcn1", "gcn2", "fc")))
            self.model.expand_adaptive_params(ctx.adj.shape[0])
        self.model.args = self.args
        ctx.logger.info("EAC task=%s nodes=%s rank=%s loss=%s prompt_only_incremental=%s",
                        self.args.year, self.model.num_nodes, self.args.rank,
                        self.args.loss, self.args.year > self.args.begin_year)

    def training_step(self, batch):
        prediction = self.predict(batch, training=True)
        kind = self.args.loss.lower()
        if kind == "mse":
            loss = F.mse_loss(prediction, batch.y, reduction="mean")
        elif kind in {"huber", "smooth_l1"}:
            loss = F.smooth_l1_loss(prediction, batch.y, reduction="mean")
        elif kind == "mae":
            loss = MAE_torch(prediction, batch.y, 0.0)
        else:
            raise ValueError("EAC loss must be mse (upstream default), huber or mae")
        return loss, {kind: loss}
