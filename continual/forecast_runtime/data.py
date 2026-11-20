"""Lazy windows matching the existing split, scale, calendar and mask protocols.

Only individual minibatches are materialized. In particular, running a new
forecaster never rewrites another model's potentially large window caches.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_madrid_shift import load_madrid_shift_source, _forward_fill, calendar_indices


SPLITS = ('train', 'val', 'test')
NORM_EPS = 1e-8


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def history_statistics(values, observed, starts, length):
    """Float64 population statistics of all history occurrences, without copies."""
    changes = np.zeros(len(values) + 1, dtype=np.int64)
    np.add.at(changes, starts, 1)
    np.add.at(changes, starts + length, -1)
    weights = np.cumsum(changes[:-1])
    denominator = 0
    total = 0.
    for lo in range(0, len(values), 256):
        hi = min(lo + 256, len(values))
        weight = weights[lo:hi, None] * observed[lo:hi]
        denominator += int(weight.sum())
        total += float((values[lo:hi].astype(np.float64) * weight).sum())
    if denominator == 0:
        raise ValueError('No observed history entries available for normalization')
    mean = total / denominator
    squares = 0.
    for lo in range(0, len(values), 256):
        hi = min(lo + 256, len(values))
        weight = weights[lo:hi, None] * observed[lo:hi]
        squares += float(((values[lo:hi].astype(np.float64) - mean) ** 2 * weight).sum())
    return {'mean': mean, 'std': max(float(np.sqrt(squares / denominator)), NORM_EPS)}


class WindowDataset(Dataset):
    def __init__(self, task, split):
        self.task = task
        self.split = split
        self.starts = task.starts[split]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        task = self.task
        start = int(self.starts[index])
        middle, end = start + 12, start + 24
        # The two float32 operations match the existing materialized loaders.
        x = task.filled[start:middle].copy()
        if task.normalize_x:
            x -= task.statistics[self.split]['mean']
            x /= task.statistics[self.split]['std']
        y = task.raw[middle:end].copy()
        valid = task.valid[middle:end].copy()
        if task.madrid and task.dst[start:end].any():
            valid[:] = False
        if task.madrid:
            y = np.where(valid, y, 0.).astype(np.float32)
        return {
            'x': torch.from_numpy(x.T.copy()),
            'y': torch.from_numpy(y.T.copy()),
            'valid': torch.from_numpy(valid.T.copy()),
            'x_tod': torch.from_numpy(task.tod[start:middle].copy()),
            'x_dow': torch.from_numpy(task.dow[start:middle].copy()),
            'y_tod': torch.from_numpy(task.tod[middle:end].copy()),
            'y_dow': torch.from_numpy(task.dow[middle:end].copy()),
        }


class TaskData:
    def __init__(self, root, config, year):
        self.year = year
        self.madrid = config['dataset'] == 'MADRID'
        self.normalize_x = config['variant'] == 'unified'
        self.steps_per_day = config['steps_per_day']
        raw_path = Path(root) / config['raw_data_path'] / f'{year}.npz'
        graph_path = Path(root) / config['graph_path'] / f'{year}_adj.npz'
        self.source_sha256 = file_hash(raw_path)
        self.graph_sha256 = file_hash(graph_path)
        if self.madrid:
            source = load_madrid_shift_source(raw_path)
            self.raw = source['x']
            self.valid = source['valid']
            self.dst = source['dst_invalid_clock']
            self.tod, self.dow = calendar_indices(source['timestamps'])
            self.node_ids = source['node_ids']
            self.active = source['active_nodes']
            self.label = str(source['timestamps'][0])[:7]
            manifest = json.loads((raw_path.parent.parent / 'manifest.json').read_text(encoding='utf-8-sig'))
            record = next(row for row in manifest['tasks'] if row['year'] == year)
            if self.source_sha256 != record['raw_file']['sha256'] or self.graph_sha256 != record['graph_file']['sha256']:
                raise ValueError('Madrid source/graph differs from the frozen selected-month manifest')
            if self.label != record['label']:
                raise ValueError('Madrid month differs from the frozen selected-month manifest')
        else:
            with np.load(raw_path, allow_pickle=False) as archive:
                self.raw = np.array(archive['x'], dtype=np.float32, copy=True)
            if config['dataset'] == 'PEMS' and config['pems_days'] > 0:
                self.raw = self.raw[:config['pems_days'] * self.steps_per_day]
            if self.raw.ndim != 2 or not np.isfinite(self.raw).all():
                raise ValueError('Source must contain a finite (time,nodes) numeric x array')
            # Evaluation excludes exact zero; training additionally applies the
            # original MAE near-zero tolerance in the objective, not here.
            self.valid = self.raw != 0
            self.dst = np.zeros(len(self.raw), dtype=bool)
            self.node_ids = np.arange(self.raw.shape[1])
            self.active = np.ones(self.raw.shape[1], dtype=bool)
            positions = np.arange(len(self.raw), dtype=np.int64)
            self.tod = positions % self.steps_per_day
            self.dow = positions // self.steps_per_day % 7
            self.label = str(year)
        with np.load(graph_path, allow_pickle=False) as archive:
            self.adj = np.array(archive['x'], dtype=np.float32, copy=True)
            if self.madrid and ('node_ids' not in archive or not np.array_equal(archive['node_ids'], self.node_ids)):
                raise ValueError('Madrid graph and source node registries differ')
        self.nodes = self.raw.shape[1]
        if self.adj.shape != (self.nodes, self.nodes) or not np.isfinite(self.adj).all() or (self.adj < 0).any():
            raise ValueError('Source/graph shape or graph weights are invalid')
        rates = (config['train_rate'], config['val_rate'], config['test_rate'])
        self.bounds = (0, int(len(self.raw) * rates[0]), int(len(self.raw) * (rates[0] + rates[1])), len(self.raw))
        if self.madrid:
            if not np.array_equal(self.active, self.valid[:self.bounds[1]].any(0)):
                raise ValueError('Madrid active nodes must be fitted on current training observations')
            self.filled, self.fallback = _forward_fill(self.raw, self.valid, self.bounds[1])
        else:
            self.filled, self.fallback = self.raw, None
        self.starts = {}
        for split, lo, hi in zip(SPLITS, self.bounds, self.bounds[1:]):
            count = hi - lo - 24
            if config['smoke']:
                count = min(count, {'train': 4, 'val': 2, 'test': 2}[split])
            if count <= 0:
                raise ValueError(f'{split} has insufficient steps for the common 12->12 protocol')
            self.starts[split] = lo + np.arange(count - 1, -1, -1, dtype=np.int64)
        self.statistics = {split: {'mean': 0., 'std': 1.} for split in SPLITS}
        if self.normalize_x:
            # Legacy preprocessing includes every history entry in its scaler,
            # including the source zeros. Madrid fits only observed entries.
            observed = self.valid if self.madrid else np.ones_like(self.valid)
            train_stats = history_statistics(self.filled, observed, self.starts['train'], 12)
            for split in SPLITS:
                self.statistics[split] = train_stats.copy() if config['normalization_scope'] == 'train' else history_statistics(self.filled, observed, self.starts[split], 12)
        self.metadata = {
            'source_sha256': self.source_sha256, 'graph_sha256': self.graph_sha256,
            'nodes': self.nodes, 'source_steps': len(self.raw), 'task_label': self.label,
            'samples': {split: len(starts) for split, starts in self.starts.items()},
            'normalize_x': self.normalize_x, 'normalization_scope': config['normalization_scope'],
            'x_statistics': self.statistics, 'target_scale': 'source',
            'window_order': 'newest_first_within_split', 'split_final_candidate': 'excluded_like_existing_psrag',
            'calendar': 'true_Monday_zero' if self.madrid else 'task_relative_seven_day_phase',
            'mask': 'explicit_original_observation_validity_true_zero_valid' if self.madrid else 'source_zero_mask',
            'storage': 'lazy_windows_no_window_cache_writes',
            'dst_excluded_windows': {split: int(sum(self.dst[s:s+24].any() for s in starts)) for split, starts in self.starts.items()} if self.madrid else {},
        }

    def context(self, device):
        stop = self.bounds[1]
        stats = self.statistics['train']
        return {'num_nodes': self.nodes, 'steps_per_day': self.steps_per_day,
                'train_raw': self.raw[:stop], 'train_valid': self.valid[:stop],
                'train_tod': self.tod[:stop], 'train_dow': self.dow[:stop],
                'input_mean': stats['mean'], 'input_std': stats['std'],
                'raw_adj': self.adj, 'node_ids': self.node_ids, 'device': device}

    def dataset(self, split):
        return WindowDataset(self, split)
