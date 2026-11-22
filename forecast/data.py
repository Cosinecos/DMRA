"""Lazy windows matching the existing split, scale, calendar and mask protocols.

Only individual minibatches are materialized. In particular, running a new
forecaster never rewrites another model's potentially large window caches.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dmra.runtime.data_madrid_shift import _forward_fill, calendar_indices, load_madrid_shift_source


SPLITS = ('train', 'val', 'test')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class WindowDataset(Dataset):
    def __init__(self, task, split):
        self.task = task
        self.starts = task.starts[split]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        task = self.task
        start = int(self.starts[index])
        middle, end = start + 12, start + 24
        x = task.filled[start:middle]
        y = task.raw[middle:end]
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
            months = source['timestamps'].astype('datetime64[M]')
            self.label = str(months[0])
            if self.label[:4] != str(year) or (months != months[0]).any():
                raise ValueError(f'Madrid source {raw_path.name} must cover one month of {year}')
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
        self.metadata = {
            'source_sha256': self.source_sha256, 'graph_sha256': self.graph_sha256,
            'nodes': self.nodes, 'source_steps': len(self.raw), 'task_label': self.label,
            'samples': {split: len(starts) for split, starts in self.starts.items()},
            'normalize_x': False, 'target_scale': 'source',
            'window_order': 'newest_first_within_split', 'split_final_candidate': 'excluded',
            'calendar': 'true_Monday_zero' if self.madrid else 'task_relative_seven_day_phase',
            'mask': 'explicit_original_observation_validity_true_zero_valid' if self.madrid else 'source_zero_mask',
            'storage': 'lazy_windows_no_window_cache_writes',
            'dst_excluded_windows': {split: int(sum(self.dst[s:s+24].any() for s in starts)) for split, starts in self.starts.items()} if self.madrid else {},
        }

    def context(self):
        return {'num_nodes': self.nodes, 'steps_per_day': self.steps_per_day,
                'raw_adj': self.adj}

    def dataset(self, split):
        return WindowDataset(self, split)
