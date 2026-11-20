"""Source windows, explicit input normalization, and validated NPZ caches.

Targets always retain the source scale. Windows retain the original project's
newest-first order and M - input_len - output_len sample count per split.
"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data


CACHE_VERSION = 1
METADATA_KEY = "__metadata__"
NORM_EPS = 1e-8
SPLITS = ("train", "val", "test")


def _bool(value):
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "yes", "y", "on"}:
            return True
        if value.strip().lower() in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")
    return bool(value)


def read_data_config(args):
    """Canonical settings checked when loading an existing window cache."""
    config = {
        "normalize_x": _bool(getattr(args, "normalize_x", False)),
        "normalization_scope": str(getattr(args, "normalization_scope", "split")).lower(),
        "x_len": int(getattr(args, "x_len", 12)),
        "y_len": int(getattr(args, "y_len", 12)),
    }
    if config["normalization_scope"] not in {"split", "train"}:
        raise ValueError("normalization_scope must be 'split' or 'train'")
    if min(config["x_len"], config["y_len"]) <= 0:
        raise ValueError("x_len and y_len must be positive")
    return config


def check_source_files(args):
    """Return missing source paths; the caller decides how to report them."""
    missing = []
    for year in range(int(args.begin_year), int(args.end_year) + 1):
        for path in (Path(args.raw_data_path) / f"{year}.npz",
                     Path(args.graph_path) / f"{year}_adj.npz"):
            if not path.is_file():
                missing.append(path)
    return missing


def source_shape(path):
    """Inspect the numeric x.npy header without materializing the source array."""
    with zipfile.ZipFile(path) as archive, archive.open("x.npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = reader(stream)
    if dtype.hasobject or len(shape) != 2:
        raise ValueError(f"Source x must be a numeric (time, nodes) array: {path}")
    return shape


def _window_split(source, offset, x_len, y_len, limit=None):
    total = x_len + y_len
    if limit is not None:
        if isinstance(limit, bool) or int(limit) != limit or int(limit) <= 0:
            raise ValueError("sample_limits values must be positive integers")
        source = source[:total + int(limit)]
    count = len(source) - total
    if count <= 0:
        raise ValueError(f"Split has {len(source)} steps; need more than {total} for a window")
    # Discard the final candidate, matching range(M-1, 0, -1) in the original.
    windows = np.lib.stride_tricks.sliding_window_view(source, total, axis=0)[:-1][::-1]
    x = np.array(windows[..., :x_len].transpose(0, 2, 1), dtype=np.float32, order="C", copy=True)
    y = np.array(windows[..., x_len:].transpose(0, 2, 1), dtype=np.float32, order="C", copy=True)
    starts = np.arange(count - 1, -1, -1, dtype=np.int64) + offset
    x_index = starts[:, None] + np.arange(x_len, dtype=np.int64)
    y_index = starts[:, None] + np.arange(x_len, total, dtype=np.int64)
    return x, y, x_index, y_index


def _edge_index(graph):
    if graph is None:
        return np.empty((2, 0), dtype=np.int64)
    if isinstance(graph, np.ndarray):
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        # Match an undirected networkx graph's one-copy edge list.
        return np.asarray(np.nonzero(np.triu(graph != 0)), dtype=np.int64)
    edges = list(graph.edges()) if callable(graph.edges) else list(graph.edges)
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2).T


def generate_samples(days, savepath, data, graph, train_rate=0.6, val_rate=0.2,
                     test_rate=0.2, val_test_mix=False, normalize_x=False,
                     normalization_scope="split", x_len=12, y_len=12,
                     sample_limits=None):
    """Create float32 windows, optionally saving one metadata-bearing cache.

    ``days`` is retained for call compatibility. The caller must truncate PEMS
    explicitly; dataset identity is never inferred from ``savepath``.
    ``sample_limits`` takes a small prefix of each original split before making
    any windows. It is intended for smoke runs and requires ``savepath=None``.
    """
    del days
    data = np.asarray(data)
    if data.ndim != 2 or data.dtype.kind not in "biuf":
        raise ValueError("Source data must be a real numeric (time, nodes) array")
    if data.shape[1] <= 0:
        raise ValueError("Source data has no nodes")
    x_len, y_len = int(x_len), int(y_len)
    if min(x_len, y_len) <= 0:
        raise ValueError("x_len and y_len must be positive")
    normalize_x = _bool(normalize_x)
    normalization_scope = str(normalization_scope).lower()
    if normalization_scope not in {"split", "train"}:
        raise ValueError("normalization_scope must be 'split' or 'train'")
    rates = (float(train_rate), float(val_rate), float(test_rate))
    if min(rates) <= 0 or not np.isclose(sum(rates), 1.0):
        raise ValueError("Positive train/val/test rates must sum to 1")
    if sample_limits is not None and savepath is not None:
        raise ValueError("Smoke sample_limits require savepath=None; do not write a partial formal cache")
    if sample_limits is not None and set(sample_limits) - set(SPLITS):
        raise ValueError("sample_limits keys must be train, val or test")

    length = data.shape[0]
    boundaries = (0, int(length * rates[0]), int(length * (rates[0] + rates[1])), length)
    result = {}
    for split, start, end in zip(SPLITS, boundaries, boundaries[1:]):
        limit = None if sample_limits is None else sample_limits.get(split)
        values = _window_split(data[start:end], start, x_len, y_len, limit)
        for suffix, value in zip(("x", "y", "x_index", "y_index"), values):
            result[f"{split}_{suffix}"] = value
    if val_test_mix:
        # Preserve the previous allocation while also shuffling time indices.
        val_size = len(result["val_x"]) if sample_limits is not None else int(length * rates[1])
        order = np.random.permutation(len(result["val_x"]) + len(result["test_x"]))
        for suffix in ("x", "y", "x_index", "y_index"):
            combined = np.concatenate((result[f"val_{suffix}"], result[f"test_{suffix}"]), axis=0)
            result[f"val_{suffix}"] = combined[order[:val_size]]
            result[f"test_{suffix}"] = combined[order[val_size:]]

    statistics = {}
    if normalize_x:
        train_mean = float(np.mean(result["train_x"], dtype=np.float64))
        train_std = max(float(np.std(result["train_x"], dtype=np.float64)), NORM_EPS)
        for split in SPLITS:
            x = result[f"{split}_x"]
            mean = train_mean if normalization_scope == "train" else float(np.mean(x, dtype=np.float64))
            std = train_std if normalization_scope == "train" else max(float(np.std(x, dtype=np.float64)), NORM_EPS)
            if not np.isfinite(mean) or not np.isfinite(std):
                raise ValueError(f"Non-finite input normalization statistics in {split}")
            x -= mean
            x /= std
            statistics[split] = {"mean": mean, "std": std}
    metadata = {
        "cache_version": CACHE_VERSION, "normalize_x": normalize_x,
        "normalization_scope": normalization_scope, "x_len": x_len, "y_len": y_len,
        "target_scale": "source", "array_dtype": "float32", "norm_eps": NORM_EPS,
        "source_steps": length, "nodes": data.shape[1], "rates": list(rates),
        "split_offsets": dict(zip(SPLITS, boundaries[:-1])),
        "split_steps": {split: end - start for split, start, end in zip(SPLITS, boundaries, boundaries[1:])},
        "val_test_mix": bool(val_test_mix), "sample_limits": sample_limits,
        "x_statistics": statistics,
    }
    result["edge_index"] = _edge_index(graph)
    result[METADATA_KEY] = np.asarray(json.dumps(metadata, sort_keys=True))
    if savepath is not None:
        path = Path(savepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **result)
    return result


def cache_metadata(inputs):
    if METADATA_KEY not in inputs:
        raise ValueError("Cache metadata is unavailable; regenerate it from source with data_process=1")
    raw = inputs[METADATA_KEY]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def load_cached_inputs(path, args):
    """Reject caches with missing metadata or incompatible normalization flags."""
    expected = read_data_config(args)
    with np.load(path, allow_pickle=False) as archive:
        metadata = cache_metadata(archive)
        if metadata.get("cache_version") != CACHE_VERSION:
            raise ValueError("Unsupported cache version; regenerate from source")
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items()
                      if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"Cache settings mismatch (saved, requested): {mismatches}; regenerate from source")
        if metadata.get("target_scale") != "source" or metadata.get("sample_limits") is not None:
            raise ValueError("Cache must contain full windows and source-scale targets")
        if bool(metadata.get("val_test_mix", False)) != _bool(getattr(args, "val_test_mix", False)):
            raise ValueError("Cache val_test_mix does not match the requested setting")
        rates = [float(getattr(args, name, default)) for name, default in
                 (("train_rate", .6), ("val_rate", .2), ("test_rate", .2))]
        if not np.allclose(metadata.get("rates", []), rates, atol=1e-12, rtol=0):
            raise ValueError("Cache split rates do not match the requested setting")
        if (hasattr(args, "expected_source_steps")
                and metadata.get("source_steps") != args.expected_source_steps):
            raise ValueError("Cache source time span does not match the requested setting; regenerate from source")
        if hasattr(args, "graph_size") and metadata.get("nodes") != args.graph_size:
            raise ValueError("Cache node count does not match the current graph")
        result = {key: archive[key] for key in archive.files}
    for split in SPLITS:
        x, y = result[f"{split}_x"], result[f"{split}_y"]
        if x.dtype != np.float32 or y.dtype != np.float32:
            raise ValueError(f"Cache {split} arrays must be float32; regenerate from source")
        if (x.ndim != 3 or y.ndim != 3 or x.shape[1] != expected["x_len"]
                or y.shape[1] != expected["y_len"] or x.shape[0] != y.shape[0]
                or x.shape[2] != y.shape[2]):
            raise ValueError(f"Invalid {split} window shapes in cache")
        for suffix, size in (("x_index", expected["x_len"]), ("y_index", expected["y_len"])):
            if result[f"{split}_{suffix}"].shape != (len(x), size):
                raise ValueError(f"Invalid {split}_{suffix} shape in cache")
    return result


class SpatioTemporalDataset(TorchDataset):
    """PyG Data batches with the original node/time orientation and metadata."""

    def __init__(self, inputs, split, x="", y="", edge_index="", mode="default", steps_per_day=288):
        self.split = split
        self.steps_per_day = int(steps_per_day)
        if self.steps_per_day <= 0:
            raise ValueError("steps_per_day must be positive")
        if mode == "default":
            self.x, self.y = inputs[split + "_x"], inputs[split + "_y"]
            self.x_index = inputs.get(split + "_x_index")
            self.y_index = inputs.get(split + "_y_index")
            train_steps = len(inputs["train_x"]) + inputs["train_x"].shape[1] + inputs["train_y"].shape[1] if "train_x" in inputs else 0
            val_steps = len(inputs["val_x"]) + inputs["val_x"].shape[1] + inputs["val_y"].shape[1] if "val_x" in inputs else 0
            self.split_offset = train_steps if split == "val" else train_steps + val_steps if split == "test" else 0
        else:
            self.x, self.y = x, y
            self.x_index = self.y_index = None
            self.split_offset = 0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        data = Data(x=torch.as_tensor(self.x[index].T, dtype=torch.float32),
                    y=torch.as_tensor(self.y[index].T, dtype=torch.float32))
        if self.x_index is None or self.y_index is None:
            x_len, y_len = self.x.shape[1], self.y.shape[1]
            end = self.split_offset + len(self.x) + x_len + y_len - 1 - int(index)
            x_index = torch.arange(end - x_len - y_len, end - y_len, dtype=torch.long)
            y_index = torch.arange(end - y_len, end, dtype=torch.long)
        else:
            x_index = torch.as_tensor(self.x_index[index], dtype=torch.long)
            y_index = torch.as_tensor(self.y_index[index], dtype=torch.long)
        data.x_tod, data.x_dow = x_index % self.steps_per_day, (x_index // self.steps_per_day) % 7
        data.y_tod, data.y_dow = y_index % self.steps_per_day, (y_index // self.steps_per_day) % 7
        return data


Dataset = SpatioTemporalDataset
