"""STBP predictor and original pattern-only incremental update protocol."""

from baselines.common import BaseAdapter
from _upstream.stbp.model import STBP_Model


DEFAULTS = {
    "dropout": 0.0,
    "model": {"in_channel": 12, "out_channel": 12, "hidden_channel": 192},
}


def defaults(dataset):
    # These are the checked-in STBP configurations, rather than PSRAG's defaults.
    if dataset == "AIR":
        return {"lr": 0.01, "batch_size": 64,
                "model": {"in_channel": 12, "out_channel": 12, "hidden_channel": 128}}
    if dataset == "PEMS":
        return {"lr": 0.01, "batch_size": 64}
    return {"lr": 0.005, "batch_size": 128}


class Adapter(BaseAdapter):
    def prepare_task(self, ctx):
        super().prepare_task(ctx)
        if self.model is None or not getattr(self.args, "init", True):
            self.args.base_node_size = ctx.adj.shape[0]
            self.model = STBP_Model(self.args).to(self.args.device)
        else:
            if ctx.adj.shape[0] < self.model.num_nodes:
                raise ValueError("STBP requires append-only node identities between tasks")
            # Original src/trainer/default_trainer.py: freeze all shared weights,
            # train existing/new patterns, and randomly initialize appended nodes.
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_("pattern_bank" in name)
            self.model.expand_adaptive_params(ctx.adj.shape[0])
        self.model.args = self.args
        ctx.logger.info("STBP task=%s nodes=%s pattern_only_incremental=%s",
                        self.args.year, self.model.num_nodes,
                        self.args.year > self.args.begin_year)
