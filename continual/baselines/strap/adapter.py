"""Full upstream STRAP (RAP) with train-only, versioned pattern libraries."""

import torch
from torch.nn import functional as F
from torch_geometric.data import Data

from baselines.common import BaseAdapter, make_loader
from _upstream.strap.model import RAP_Model
from _upstream.strap.ewc import EWC


DEFAULTS = {
    "lr": 0.03, "batch_size": 128, "epoch": 100, "patience": 5,
    "dropout": 0.0, "loss": "mse", "strategy": "retrain", "init": True,
    "gcn": {"in_channel": 12, "out_channel": 12, "hidden_channel": 64},
    "tcn": {"in_channel": 1, "out_channel": 1, "kernel_size": 3, "dilation": 1},
    "backbone_type": "stgnn", "use_strap": True,
    "ewc": False, "ewc_lambda": 0.0005, "ewc_strategy": "ewc",
    "k_neighbors": 50, "history_ratio": 0.3, "history_limit": 10000,
    "retrieval_batch_size": 128, "retrieval_key_batch_size": 4096,
    "fusion_weight": 0.7, "return_pattern_or_value": "value", "projection_dim": 64,
    "use_spatial_lib": True, "use_temporal_lib": True, "use_spatiotemporal_lib": True,
    "spatial_dropout": 0.0, "temporal_dropout": 0.0, "spatiotemporal_dropout": 0.0,
    "spatial_retrieval_count": 50000, "temporal_retrieval_count": 50000,
    "spatiotemporal_retrieval_count": 100000, "temporal_k_hop": 2,
    "spatial_clusters": None, "max_neighbors": 20, "max_cluster_nodes": 100,
    "time_window": 12, "time_overlap": 6, "cross_pattern_count": 2000,
    "curvature_pattern_count": 2000, "pattern_build_samples": 128,
}


def defaults(dataset):
    return {"dropout": 0.1 if dataset == "ENERGY_WIND" else 0.0}


class Adapter(BaseAdapter):
    def __init__(self, args):
        super().__init__(args)
        self.ewc = None

    def prepare_task(self, ctx):
        super().prepare_task(ctx)
        self.args = ctx.args
        args = self.args
        continuing = self.model is not None and args.init
        if not continuing:
            self.model = RAP_Model(args).to(args.device)
        self.model.args = args
        self.model.current_year = args.year
        if self.model.use_strap:
            count = int(args.pattern_build_samples)
            # Windows are newest first; only current training data builds the bank.
            features = torch.as_tensor(ctx.inputs["train_x"][:count], dtype=torch.float32,
                                       device=args.device).flip(0).transpose(1, 2).contiguous()
            self.model.initialize_patterns(Data(x=features), ctx.adj)
            args.logger.info("STRAP training-only library task=%s samples=%s sources=%s",
                             args.year, len(features), self.model.strap.source_counts)
        self.ewc = None
        if continuing and args.ewc:
            self.ewc = EWC(self.model, ctx.adj, args.ewc_lambda, args.ewc_strategy)
            self.ewc.register_ewc_params(make_loader(ctx.inputs, "train", args),
                                       self._loss, args.device)

    def _loss(self, pred, target, reduction="mean"):
        if self.args.loss == "mse":
            return F.mse_loss(pred, target, reduction=reduction)
        if self.args.loss == "huber":
            return F.smooth_l1_loss(pred, target, reduction=reduction)
        if self.args.loss == "mae":
            return F.l1_loss(pred, target, reduction=reduction)
        raise ValueError("STRAP loss must be mse, huber or mae")

    def training_step(self, batch):
        pred = self.predict(batch, training=True)
        prediction = self._loss(pred, batch.y)
        consolidation = (self.ewc.compute_consolidation_loss() if self.ewc is not None
                         else prediction.new_zeros(()))
        return prediction + consolidation, {"prediction": prediction, "ewc": consolidation}

    def extra_state(self):
        result = {}
        if self.model is not None and self.model.use_strap:
            strap = self.model.strap
            result["strap"] = {
                "current_year": self.model.current_year,
                "source_counts": strap.source_counts,
                "banks": {kind: {name: getattr(bank, name) for name in ("keys", "values", "topology")}
                          for kind, bank in strap.banks.items()},
            }
        if self.ewc is not None:
            result["ewc_buffers"] = {name: value for name, value in self.ewc.named_buffers()
                                     if "_estimated_" in name}
        return result

    def load_extra_state(self, state):
        if "strap" in state:
            payload = state["strap"]
            device = self.model.strap.projector.weight.device
            libraries = {kind: {name: value.to(device) for name, value in fields.items()}
                         for kind, fields in payload["banks"].items()}
            self.model.current_year = payload["current_year"]
            self.model.strap._activate(libraries, payload["current_year"], payload["source_counts"])
        if self.ewc is not None:
            for name, value in state.get("ewc_buffers", {}).items():
                setattr(self.ewc, name, value.to(self.args.device))
