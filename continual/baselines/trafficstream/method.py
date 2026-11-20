"""Selection and EWC used by the TrafficStream/PECPM source ports.

There is no historical-sample replay: upstream 'replay' selects stable/random
nodes whose CURRENT training windows are included in the incremental subgraph.
All statistics here use training data. Future validation/test data never enter
the feature drift, daily profiles, representative patterns, or Fisher estimate.
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from baselines.common import make_loader


def source_subgraph(raw_adj, seeds, hops):
    """Keep the authors' single-copy NetworkX edges and PyG default flow.

    nx.Graph merges nonzero raw edges from either direction, but its edges()
    list contains each edge only once. Upstream feeds that list directly to
    k_hop_subgraph without to_undirected(): source_to_target therefore expands
    predecessors on that single-copy orientation. Preserve this surprising
    source behavior instead of substituting an undirected breadth-first walk.
    """
    size = raw_adj.shape[0]
    if not size:
        raise ValueError('A training subgraph requires at least one source node')
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    seeds = seeds[(seeds >= 0) & (seeds < size)]
    if not len(seeds):
        # Explicit tiny/empty-selection guard; the five real streams select
        # nonempty seed sets. Keep the pipeline executable on an edgeless toy.
        seeds = np.arange(size, dtype=np.int64)
    graph = nx.from_numpy_array(raw_adj)
    edges = np.asarray(list(graph.edges), dtype=np.int64).reshape(-1, 2).T
    edge_index = torch.as_tensor(edges, dtype=torch.long)
    participating = np.unique(edges)
    filtered = seeds[np.isin(seeds, participating)]
    if len(filtered):
        seeds = filtered
    # Source drops isolated seeds before calling PyG. If this would remove
    # EVERY requested seed, retain those seeds as a documented toy guard: its
    # predictor can still use its self path on a zero-edge subgraph.
    subset, sub_edges, mapping, _ = k_hop_subgraph(
        seeds.tolist(), num_hops=int(hops), edge_index=edge_index,
        relabel_nodes=True, num_nodes=size)
    return subset, sub_edges, mapping, seeds


def node_subgraph(inputs, raw_adj, seeds, hops, device):
    """Native PyG-selected nodes with unweighted, symmetric train adjacency."""
    source_nodes, sub_edges, mapping, _ = source_subgraph(raw_adj, seeds, hops)
    subset = source_nodes.numpy()
    selected = dict(inputs)
    for split in ('train', 'val', 'test'):
        for suffix in ('x', 'y'):
            selected[f'{split}_{suffix}'] = inputs[f'{split}_{suffix}'][..., subset]
    # Exact source train() reconstruction: weights do not travel in edge_index.
    train_graph = nx.Graph()
    train_graph.add_nodes_from(range(len(subset)))
    train_graph.add_edges_from(sub_edges.numpy().T)
    binary = nx.to_numpy_array(train_graph)
    binary /= binary.sum(axis=1, keepdims=True) + 1e-6
    adjacency = torch.as_tensor(binary, dtype=torch.float32, device=device)
    return selected, adjacency, subset, mapping.to(device)


def training_daily(raw_data, steps_per_day, train_rate=.6):
    """C[i,tod] = mean of observed training values at time-of-day tod."""
    train = np.asarray(raw_data[:int(len(raw_data) * train_rate)], dtype=np.float32)
    if not len(train):
        raise ValueError('Daily statistics require a nonempty training segment')
    slots = np.arange(len(train)) % int(steps_per_day)
    counts = np.bincount(slots, minlength=int(steps_per_day))
    sums = np.zeros((int(steps_per_day), train.shape[1]), dtype=np.float64)
    np.add.at(sums, slots, train)
    means = sums / np.maximum(counts[:, None], 1)
    missing = counts == 0
    if missing.any():
        means[missing] = train.mean(axis=0)
    return means.T.astype(np.float32)


def recent_features(model, inputs, adj, args):
    """Nonoverlapping windows over the latest seven TRAINING days."""
    windows = inputs['train_x']
    count = max(1, int(7 * args.steps_per_day) // int(args.x_len))
    # Source-compatible generated windows are newest-first. Index stride L
    # yields the same nonoverlapping extraction used by detect.get_feature.
    selected = windows[:count * args.x_len:args.x_len]
    features = []
    with torch.no_grad():
        for start in range(0, len(selected), max(1, int(args.batch_size))):
            signal = torch.as_tensor(selected[start:start + args.batch_size].transpose(0, 2, 1),
                                     device=args.device, dtype=torch.float32)
            data = Data(x=signal.reshape(-1, args.x_len))
            output = model.feature(data, adj)
            features.append(output.reshape(len(signal), adj.shape[0], -1).cpu().numpy())
    return np.concatenate(features, axis=0).transpose(1, 0, 2)


def feature_js(previous, current, bins=10):
    """Sum the per-feature Jensen-Shannon distances, as upstream detect.py."""
    def probabilities(features):
        low = features.min(axis=1, keepdims=True)
        span = features.max(axis=1, keepdims=True) - low
        normalized = np.divide(features - low, span, out=np.zeros_like(features), where=span > 0)
        # Use the exact dtype-specific linspace boundaries from np.histogram.
        # Multiplying by bins can round a value immediately below an edge up
        # into the next bin (e.g. float64 0.6 vs linspace's 0.6000000000000001).
        edges = np.linspace(0, 1, bins + 1, dtype=normalized.dtype)
        assignments = np.minimum(np.searchsorted(edges, normalized, side='right') - 1, bins - 1)
        # Separate histogram for every node and feature, avoiding Python N*D loops.
        assignments = assignments.transpose(0, 2, 1)
        dimensions = assignments.shape[:2]
        offsets = np.arange(np.prod(dimensions)).reshape(*dimensions, 1) * bins
        counts = np.bincount((assignments + offsets).ravel(), minlength=np.prod(dimensions) * bins)
        counts = counts.reshape(*dimensions, bins).astype(np.float64)
        return counts / np.maximum(counts.sum(-1, keepdims=True), 1)

    common = min(len(previous), len(current))
    p, q = probabilities(previous[:common]), probabilities(current[:common])
    midpoint = (p + q) / 2
    def kl(prob):
        ratio = np.divide(prob, midpoint, out=np.ones_like(prob), where=midpoint > 0)
        return np.where(prob > 0, prob * np.log(np.maximum(ratio, 1e-300)), 0).sum(-1)
    return np.sqrt(np.maximum((kl(p) + kl(q)) / 2, 0)).sum(-1)


class FisherPenalty:
    """Upstream EWC: summed squared batch gradients and theta reference.

    Deliberately retains the source's sum rather than changing it to a batch
    average. The estimate is made with the previous model and the current
    task's training data, exactly as main.py does in the public repositories.
    """

    def __init__(self):
        self.fisher = {}
        self.means = {}

    def estimate(self, adapter, inputs, prediction_loss, max_batches=0):
        args = adapter.args
        parameters = [(name, value) for name, value in adapter.model.named_parameters() if value.requires_grad]
        self.means = {name: value.detach().clone() for name, value in parameters}
        self.fisher = {name: torch.zeros_like(value) for name, value in parameters}
        adapter.model.eval()
        loader = make_loader(inputs, 'train', args, shuffle=False)
        batches = 0
        for batch in loader:
            batch = batch.to(args.device)
            with torch.enable_grad():
                prediction = adapter.predict(batch, training=False)
                loss = prediction_loss(prediction, batch.y)
                gradients = torch.autograd.grad(loss, [value for _, value in parameters], allow_unused=True)
            for (name, _), gradient in zip(parameters, gradients):
                if gradient is not None:
                    self.fisher[name].add_(gradient.detach().square())
            batches += 1
            if max_batches and batches >= max_batches:
                break
        adapter.model.zero_grad(set_to_none=True)
        if any(not torch.isfinite(value).all() for value in self.fisher.values()):
            raise FloatingPointError('Non-finite Fisher information; reduce the learning rate or rescale the data explicitly')
        args.logger.info('EWC Fisher batches=%s elements=%s', batches, sum(x.numel() for x in self.fisher.values()))

    def loss(self, model, strength):
        total = next(model.parameters()).new_zeros(())
        for name, parameter in model.named_parameters():
            if name in self.fisher:
                total = total + (self.fisher[name] * (parameter - self.means[name]).square()).sum()
        return (float(strength) / 2) * total

    def state(self):
        return {'fisher': self.fisher, 'means': self.means}

    def load(self, state, device):
        self.fisher = {name: value.to(device) for name, value in state.get('fisher', {}).items()}
        self.means = {name: value.to(device) for name, value in state.get('means', {}).items()}
