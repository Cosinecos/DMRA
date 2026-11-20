"""STKEC: long-term cluster memory, subgraph expansion and Fisher EWC.

The upstream undefined-input/day-vector/argmax/detached-attention defects are
repaired explicitly; prediction equations in STKEC_model.py are unchanged.
"""

import random

from Bio.Cluster import kcluster
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.utils import k_hop_subgraph

from madrid_continual_runtime.common import BaseAdapter, make_loader, masked_loss
from madrid_continual_runtime.baselines.trafficstream.method import observable_seeds
from _upstream.stkec.STKEC_model import Basic_Model
from _upstream.stkec.stkec_ewc import EWC


DEFAULTS = {
    "lr": 0.01, "batch_size": 64, "epoch": 100, "patience": 8,
    "dropout": 0.0, "loss": "mse", "init": True,
    "strategy": "incremental", "increase": True, "num_hops": 2,
    "cluster": 24, "ewc": True, "ewc_lambda": 0.0005, "ewc_strategy": "ewc",
    "minimum_node_ratio": 0.2, "cluster_loss_weight": 0.1,
    "gcn": {"in_channel": 12, "out_channel": 12, "hidden_channel": 64},
    "tcn": {"in_channel": 1, "out_channel": 1, "kernel_size": 3, "dilation": 1},
}


def defaults(dataset):
    # Upstream only provides PEMS. Other datasets retain the same method settings.
    return {}


def undirected_edge_index(adjacency):
    """Match nx.from_numpy_array(A, create_using=Graph).edges for asymmetric A."""
    adjacency = np.asarray(adjacency)
    present = (adjacency != 0) | (adjacency.T != 0)
    row, col = np.nonzero(np.triu(present))
    return torch.as_tensor(np.stack((row, col)), dtype=torch.long)


def long_term_labels(training_data, steps_per_day, cluster_count):
    """Aggregate complete training days into a node-by-time-of-day matrix."""
    training_data = np.asarray(training_data, dtype=np.float64)
    if training_data.ndim != 2 or not len(training_data):
        raise ValueError("STKEC clustering requires nonempty training-only [time,node] data")
    steps_per_day = int(steps_per_day)
    days = len(training_data) // steps_per_day
    if days:
        daily = training_data[:days * steps_per_day].reshape(days, steps_per_day, -1).sum(0).T
    else:
        # No full day: cluster the available chronological training vector.
        daily = training_data.T.copy()
    count = min(int(cluster_count), daily.shape[0])
    if count < 1:
        raise ValueError("STKEC cluster must be positive")
    if count == 1:
        return np.zeros(len(daily), dtype=np.int64)
    # Bio.Cluster's private random initializer is not controlled by NumPy's seed.
    # Supply a seeded initial partition so the project seed controls this step.
    initial = np.arange(len(daily), dtype=np.int32) % count
    np.random.shuffle(initial)
    labels, _, _ = kcluster(np.ascontiguousarray(daily), nclusters=count,
                            dist="u", initialid=initial)
    return np.asarray(labels, dtype=np.int64)


class Adapter(BaseAdapter):
    def __init__(self, args):
        super().__init__(args)
        self.ewc = None
        self.mapping = None
        self.labels = None
        self.selected_nodes = None
        self.subgraph_nodes = None

    def prepare_task(self, ctx):
        super().prepare_task(ctx)
        self.args = ctx.args
        args = self.args
        continuing = self.model is not None and args.init
        self.mapping = None
        self.ewc = None
        if not continuing:
            self.model = Basic_Model(args).to(args.device)
            if str(args.dataset).startswith("MADRID"):
                active = np.asarray(ctx.active_nodes, dtype=bool)
                values = np.asarray(ctx.train_raw_data)
                valid = getattr(ctx, "train_raw_valid", None)
                if valid is not None:
                    values = np.where(valid, values, 0)
                labels = np.zeros(values.shape[1], dtype=np.int64)
                labels[active] = long_term_labels(values[:, active], args.steps_per_day, args.cluster)
            else:
                labels = long_term_labels(ctx.train_raw_data, args.steps_per_day, args.cluster)
            self.labels = torch.as_tensor(labels, device=args.device, dtype=torch.long)
            args.logger.info("STKEC long-term clusters use only training data: steps=%s day=%s classes=%s",
                             len(ctx.train_raw_data), args.steps_per_day, len(np.unique(labels)))
        self.model.args = args
        count = ctx.adj.shape[0]
        if continuing and args.ewc:
            # Official STKEC estimates Fisher on current-task full training data
            # using the previous best model before adapting this year's subgraph.
            self.ewc = EWC(self.model, ctx.adj, args.ewc_lambda, args.ewc_strategy)
            self.ewc.register_ewc_params(make_loader(ctx.inputs, "train", args), self._loss, args.device)
        if continuing and args.strategy == "incremental":
            old_count = ctx.previous_raw_adj.shape[0]
            selected = list(range(old_count, count)) if args.increase else []
            minimum = min(count, max(1, int(float(args.minimum_node_ratio) * count)))
            if len(selected) < minimum:
                remaining = sorted(set(range(count)) - set(selected))
                selected.extend(random.sample(remaining, minimum - len(selected)))
            selected = observable_seeds(ctx, selected)
            # Match the public main.py's undirected, single-copy NetworkX edge list.
            edges = undirected_edge_index(ctx.raw_adj)
            connected = set(edges[0].tolist()) | set(edges[1].tolist())
            selected = sorted(set(selected) & connected)
            if not selected:
                raise ValueError("STKEC selected no connected nodes; cannot train an empty incremental subgraph")
            nodes, sub_edges, mapping, _ = k_hop_subgraph(
                selected, num_hops=int(args.num_hops), edge_index=edges,
                relabel_nodes=True, num_nodes=count)
            indices = nodes.numpy()
            self.train_inputs = dict(ctx.inputs)
            # Only training is sliced; common validation/test evaluate the full graph.
            for suffix in ("x", "y", "x_valid", "y_valid"):
                key = "train_" + suffix
                if key in ctx.inputs:
                    self.train_inputs[key] = ctx.inputs[key][:, :, indices]
            for key in ("node_ids", "active_nodes"):
                if key in ctx.inputs:
                    self.train_inputs[key] = np.asarray(ctx.inputs[key])[indices]
            adjacency = torch.zeros(len(nodes), len(nodes), device=args.device)
            edge_index = sub_edges.to(args.device)
            adjacency[edge_index[0], edge_index[1]] = 1
            adjacency[edge_index[1], edge_index[0]] = 1
            self.train_adj = adjacency / (adjacency.sum(1, keepdim=True) + 1e-6)
            self.mapping = mapping.to(args.device)
            self.selected_nodes = torch.tensor(selected, dtype=torch.long)
            self.subgraph_nodes = nodes
            args.logger.info("STKEC incremental selected=%s subgraph=%s total=%s Fisher=%s",
                             len(selected), len(nodes), count, self.ewc is not None)

    def _loss(self, pred, target, reduction="mean", valid=None):
        if str(self.args.dataset).startswith("MADRID"):
            return masked_loss(pred, target, valid, kind=self.args.loss,
                               args=self.args, reduction=reduction)
        if self.args.loss == "mse":
            return F.mse_loss(pred, target, reduction=reduction)
        if self.args.loss == "huber":
            return F.smooth_l1_loss(pred, target, reduction=reduction)
        if self.args.loss == "mae":
            return F.l1_loss(pred, target, reduction=reduction)
        raise ValueError("STKEC loss must be mse, huber or mae")

    def predict(self, batch, training=False):
        pred, _ = self.model(batch, self.train_adj if training else self.context.adj)
        return pred

    def training_step(self, batch):
        pred, attention = self.model(batch, self.train_adj)
        truth = batch.y
        valid = getattr(batch, "y_valid", None)
        if self.mapping is not None:
            nodes = self.train_adj.shape[0]
            pred = pred.reshape(-1, nodes, pred.shape[-1])[:, self.mapping]
            truth = truth.reshape(-1, nodes, truth.shape[-1])[:, self.mapping]
            if valid is not None:
                valid = valid.reshape(-1, nodes, truth.shape[-1])[:, self.mapping]
        prediction = self._loss(pred, truth, valid=valid)
        cluster = prediction.new_zeros(())
        if self.args.year == self.args.begin_year:
            batch_count = attention.shape[0] // len(self.labels)
            # Upstream .data.cpu() disconnected the intended memory supervision;
            # max(one-hot) also returned class 1 for every node. Use true IDs.
            labels = self.labels.repeat(batch_count)
            if str(self.args.dataset).startswith("MADRID"):
                active = torch.as_tensor(self.context.active_nodes, device=attention.device, dtype=torch.bool)
                keep = active.repeat(batch_count)
                cluster = F.cross_entropy(attention[keep], labels[keep]) if keep.any() else attention.sum()*0
            else:
                cluster = F.cross_entropy(attention, labels)
        consolidation = (self.ewc.compute_consolidation_loss() if self.ewc is not None
                         else prediction.new_zeros(()))
        loss = prediction + float(self.args.cluster_loss_weight) * cluster + consolidation
        return loss, {"prediction": prediction, "cluster": cluster, "ewc": consolidation}

    def extra_state(self):
        result = {"labels": self.labels, "mapping": self.mapping,
                  "selected_nodes": self.selected_nodes, "subgraph_nodes": self.subgraph_nodes}
        if self.ewc is not None:
            result["ewc_buffers"] = {name: value for name, value in self.ewc.named_buffers()
                                     if "_estimated_" in name}
        return result

    def load_extra_state(self, state):
        self.labels = state.get("labels")
        self.mapping = state.get("mapping")
        if self.labels is not None:
            self.labels = self.labels.to(self.args.device)
        if self.mapping is not None:
            self.mapping = self.mapping.to(self.args.device)
        self.selected_nodes = state.get("selected_nodes")
        self.subgraph_nodes = state.get("subgraph_nodes")
        if self.ewc is not None:
            for name, value in state.get("ewc_buffers", {}).items():
                setattr(self.ewc, name, value.to(self.args.device))
