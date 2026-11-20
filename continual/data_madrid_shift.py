"""Independent Madrid data adapter with explicit validity and real local clocks.

Legal zero flow remains valid. Inputs are forward-filled in chronological order;
the initial fallback is a per-node median fitted on the training segment only.
That fallback is an offline training-fitted imputer, not a strictly online
imputer within the training segment. Validation/test never supply its fit data.

Source timestamps use a nominal local quarter-hour clock. The last Sunday of
March/October at02:00-02:45 is explicitly invalid, and any sample whose history
or target touches those four slots has its complete target mask disabled.
"""
from __future__ import annotations

import json
import calendar
from pathlib import Path
import re

import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from data import _bool, _edge_index, _window_split, cache_metadata


ADAPTER = "madrid_selected_month15_mmd_v1"
CACHE_VERSION = 1
METADATA_KEY = "__metadata__"
SPLITS = ("train", "val", "test")
NORM_EPS = 1e-8
STEP_NS = 15 * 60 * 1_000_000_000
DST_POLICY = "exclude_whole_window_when_history_or_target_touches_last_Sunday_March_October_local_02_00_to_02_45"


def _mask(values, shape, name):
    values = np.asarray(values)
    if values.shape != tuple(shape) or values.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a boolean/0-1 array with shape {tuple(shape)}")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must contain only finite boolean/0-1 values")
    return values.astype(bool, copy=True)


def validate_node_ids(node_ids, nodes=None):
    values = np.asarray(node_ids)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("node_ids must be a one-dimensional integer array")
    if nodes is not None and len(values) != int(nodes):
        raise ValueError("node_ids length does not match the source node count")
    if not len(values) or (values < 0).any() or len(np.unique(values)) != len(values):
        raise ValueError("node_ids must be nonempty, unique and nonnegative")
    return values.astype(np.int64, copy=True)


def ensure_node_prefix(previous_ids, current_ids):
    """Reject removal/reordering that would silently misassign old embeddings."""
    previous = validate_node_ids(previous_ids)
    current = validate_node_ids(current_ids)
    if len(current) < len(previous) or not np.array_equal(current[:len(previous)], previous):
        raise ValueError("Madrid node growth requires previous node_ids as an unchanged prefix")
    return len(current) - len(previous)


def validate_timestamps(timestamps, steps=None):
    values = np.asarray(timestamps)
    if values.ndim != 1 or (steps is not None and len(values) != int(steps)):
        raise ValueError("timestamps must contain one timestamp per source time row")
    if values.dtype.kind in "iu":
        clock = values.astype(np.int64).astype("datetime64[ns]")
    elif values.dtype.kind == "M":
        clock = values.astype("datetime64[ns]")
    elif values.dtype.kind in "US":
        strings = values.astype(str)
        if any(re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", item) for item in strings):
            raise ValueError("timestamps must be timezone-free Madrid local wall-clock values")
        try:
            clock = strings.astype("datetime64[ns]")
        except ValueError as exc:
            raise ValueError("timestamps contain an invalid local ISO datetime") from exc
    else:
        raise ValueError("timestamps must be int64 local nanoseconds, datetime64 or ISO Unicode")
    if not len(clock) or np.isnat(clock).any():
        raise ValueError("timestamps must be nonempty and cannot contain NaT")
    ticks = clock.astype(np.int64)
    if np.any(ticks % STEP_NS):
        raise ValueError("timestamps must align to a 15-minute local time grid")
    if len(ticks) > 1 and not np.all(np.diff(ticks) == STEP_NS):
        raise ValueError("timestamps must be consecutive unique 15-minute local observations")
    return clock


def calendar_indices(timestamps):
    clock = validate_timestamps(timestamps)
    minutes = clock.astype("datetime64[m]").astype(np.int64)
    days = clock.astype("datetime64[D]").astype(np.int64)
    tod = (minutes % (24 * 60)) // 15
    # 1970-01-01 was Thursday; use Monday=0, Sunday=6.
    dow = (days + 3) % 7
    return tod.astype(np.int64), dow.astype(np.int64)


def dst_clock_mask(timestamps):
    """Conservative invalid local clock slots, independent of observed values."""
    clock=validate_timestamps(timestamps)
    result=np.zeros(len(clock),dtype=bool)
    for year_value in np.unique(clock.astype("datetime64[Y]")):
        year=int(str(year_value))
        for month in (3,10):
            last=calendar.monthrange(year,month)[1]
            day=last-(calendar.weekday(year,month,last)+1)%7
            start=np.datetime64(f"{year}-{month:02d}-{day:02d}T02:00:00","ns")
            result |= (clock>=start)&(clock<start+np.timedelta64(1,"h"))
    return result


def _validated_source(data, valid, timestamps, node_ids=None, coordinates=None, active_nodes=None, dst_invalid_clock=None):
    values = np.asarray(data)
    if values.ndim != 2 or values.dtype.kind not in "biuf" or values.shape[1] <= 0:
        raise ValueError("Madrid source x must be a real numeric (time, nodes) matrix")
    mask = _mask(valid, values.shape, "valid")
    if not np.isfinite(values[mask]).all() or (values[mask] < 0).any():
        raise ValueError("Madrid valid observations must be finite nonnegative flow; zero is valid")
    clock = validate_timestamps(timestamps, values.shape[0])
    expected_dst=dst_clock_mask(clock)
    if dst_invalid_clock is not None:
        actual_dst=_mask(dst_invalid_clock,(len(clock),),"dst_invalid_clock")
        if not np.array_equal(actual_dst,expected_dst):
            raise ValueError("Madrid DST flags do not match the declared local clock rule")
    if mask[expected_dst].any():
        raise ValueError("Known Madrid DST clock slots cannot contain valid observations")
    ids = validate_node_ids(np.arange(values.shape[1], dtype=np.int64) if node_ids is None else node_ids, values.shape[1])
    # Graph availability is fitted on the training segment. A previously known
    # node can have no training observations and return during validation/test;
    # its genuine labels must remain valid even while its graph row is inactive.
    result = dict(x=values.astype(np.float32, copy=True), valid=mask, timestamps=clock,
                  node_ids=ids,dst_invalid_clock=expected_dst)
    if active_nodes is not None:
        result["active_nodes"] = _mask(active_nodes, (values.shape[1],), "active_nodes")
    if coordinates is not None:
        coords = np.asarray(coordinates)
        if coords.shape != (values.shape[1], 2) or coords.dtype.kind not in "iuf" or not np.isfinite(coords).all():
            raise ValueError("coordinates must be a finite numeric (nodes, 2) array")
        result["coordinates"] = coords.astype(np.float64, copy=True)
    return result


def load_madrid_shift_source(path):
    """Load strict, pickle-free prepared Madrid source arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        # Prepared source availability must be explicit, since inferring it from
        # the whole month would reveal a test-only returning node before train.
        missing = {"x", "valid", "timestamps", "node_ids", "active_nodes", "coordinates", "dst_invalid_clock"} - set(archive.files)
        if missing:
            raise ValueError(f"Madrid source is missing required arrays: {sorted(missing)}")
        result = _validated_source(archive["x"], archive["valid"], archive["timestamps"],
                                   node_ids=archive["node_ids"],
                                   coordinates=archive["coordinates"] if "coordinates" in archive else None,
                                   active_nodes=archive["active_nodes"] if "active_nodes" in archive else None,
                                   dst_invalid_clock=archive["dst_invalid_clock"])
        if METADATA_KEY in archive:
            result[METADATA_KEY] = archive[METADATA_KEY].copy()
    return result


def _forward_fill(values, valid, train_stop):
    fallback = np.zeros(values.shape[1], dtype=np.float32)
    for node in range(values.shape[1]):
        observed = values[:train_stop, node][valid[:train_stop, node]]
        if len(observed):
            fallback[node] = np.median(observed)
    last = fallback.copy()
    filled = np.empty_like(values, dtype=np.float32)
    for index in range(len(values)):
        last = np.where(valid[index], values[index], last)
        filled[index] = last
    if not np.isfinite(filled).all():
        raise ValueError("Forward-filled Madrid inputs are not finite float32")
    return filled, fallback


def generate_madrid_shift_samples(data, valid, timestamps, graph=None, node_ids=None, *,
                            train_rate=.6, val_rate=.2, test_rate=.2,
                            x_len=12, y_len=12, normalize_x=False,
                            normalization_scope="train", sample_limits=None,
                            savepath=None, coordinates=None, active_nodes=None,
                            source_sha256=None,dst_invalid_clock=None):
    """Build newest-first within-split windows with independent source validity.

    Input normalization statistics use observed history entries only, including
    real zeros; imputed entries never fit the scaler. In ``split`` mode the
    validation/test input statistics are explicitly fitted on their own inputs,
    matching that requested protocol. Targets always retain source units.
    """
    source = _validated_source(data, valid, timestamps, node_ids, coordinates, active_nodes,dst_invalid_clock)
    values, mask, clock = source["x"], source["valid"], source["timestamps"]
    if isinstance(graph, np.ndarray) and graph.shape != (values.shape[1], values.shape[1]):
        raise ValueError("Madrid graph must match the source node count")
    if isinstance(x_len, bool) or isinstance(y_len, bool) or int(x_len) != x_len or int(y_len) != y_len:
        raise ValueError("x_len and y_len must be positive integers")
    x_len, y_len = int(x_len), int(y_len)
    if min(x_len, y_len) <= 0:
        raise ValueError("x_len and y_len must be positive integers")
    rates = tuple(map(float, (train_rate, val_rate, test_rate)))
    if not np.isfinite(rates).all() or min(rates) <= 0 or not np.isclose(sum(rates), 1):
        raise ValueError("Positive train/val/test rates must sum to 1")
    normalize_x = _bool(normalize_x)
    normalization_scope = str(normalization_scope).lower()
    if normalization_scope not in {"train", "split"}:
        raise ValueError("normalization_scope must be train or split")
    if sample_limits is not None and savepath is not None:
        raise ValueError("Smoke sample_limits require savepath=None")
    if sample_limits is not None and set(sample_limits) - set(SPLITS):
        raise ValueError("sample_limits keys must be train, val or test")
    if source_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}",str(source_sha256)):
        raise ValueError("source_sha256 must be the prepared source file's 64-character SHA256")
    bounds = (0, int(len(values)*rates[0]), int(len(values)*(rates[0]+rates[1])), len(values))
    active = source.get("active_nodes")
    if active is None:
        active = mask[:bounds[1]].any(axis=0)
    filled, fallback = _forward_fill(values, mask, bounds[1])
    tod, dow = calendar_indices(clock)
    dst=source["dst_invalid_clock"]
    result = dict(node_ids=source["node_ids"], active_nodes=active,dst_invalid_clock=dst)
    excluded_counts={}
    if "coordinates" in source:
        result["coordinates"] = source["coordinates"]
    for split, start, end in zip(SPLITS, bounds, bounds[1:]):
        limit = None if sample_limits is None else sample_limits.get(split)
        x, _, x_index, y_index = _window_split(filled[start:end], start, x_len, y_len, limit)
        excluded=dst[x_index].any(1)|dst[y_index].any(1)
        target_valid = mask[y_index].copy()
        target_valid[excluded]=False
        excluded_counts[split]=int(excluded.sum())
        y = np.where(target_valid, values[y_index], 0).astype(np.float32)
        result.update({f"{split}_x": x, f"{split}_y": y,
                       f"{split}_x_index": x_index, f"{split}_y_index": y_index,
                       f"{split}_x_valid": mask[x_index].copy(), f"{split}_y_valid": target_valid.copy(),
                       f"{split}_dst_excluded":excluded,
                       f"{split}_x_tod": tod[x_index], f"{split}_x_dow": dow[x_index],
                       f"{split}_y_tod": tod[y_index], f"{split}_y_dow": dow[y_index]})
    statistics = {}
    if normalize_x:
        observed = result["train_x"][result["train_x_valid"]]
        if not len(observed):
            raise ValueError("No valid training history observations to fit input normalization")
        train_mean = float(np.mean(observed, dtype=np.float64))
        train_std = max(float(np.std(observed, dtype=np.float64)), NORM_EPS)
        for split in SPLITS:
            observed = result[f"{split}_x"][result[f"{split}_x_valid"]]
            if normalization_scope == "split" and not len(observed):
                raise ValueError(f"No valid {split} histories for split normalization")
            mean = train_mean if normalization_scope == "train" else float(np.mean(observed, dtype=np.float64))
            std = train_std if normalization_scope == "train" else max(float(np.std(observed, dtype=np.float64)), NORM_EPS)
            result[f"{split}_x"] -= mean
            result[f"{split}_x"] /= std
            statistics[split] = dict(mean=mean, std=std)
    metadata = dict(adapter=ADAPTER, cache_version=CACHE_VERSION, normalize_x=normalize_x,
                    normalization_scope=normalization_scope, x_len=x_len, y_len=y_len,
                    steps_per_day=96, timestamp_unit="local_wall_clock_nanoseconds", weekday="Monday=0",
                    first_timestamp=str(clock[0]), last_timestamp=str(clock[-1]),
                    source_steps=len(values), nodes=values.shape[1], rates=list(rates),
                    split_offsets=dict(zip(SPLITS, bounds[:-1])),
                    split_steps={s: end-start for s,start,end in zip(SPLITS,bounds,bounds[1:])},
                    sample_limits=sample_limits, val_test_mix=False,
                    target_scale="source", array_dtype="float32", zero_is_valid=True,
                    mask_policy="explicit_original_observation_validity",
                    input_fill="chronological_forward_fill_with_training_fitted_node_median_initial_fallback",
                    training_fitted_fallback=fallback.tolist(), x_statistics=statistics,
                    normalization_fit="observed_history_entries_only_including_real_zeros",
                    source_valid_observations=int(mask.sum()),
                    source_valid_zeros=int(((values == 0) & mask).sum()),
                    graph_availability_default="training_observations_only",
                    dst_policy=DST_POLICY,dst_invalid_clock_indices=np.flatnonzero(dst).tolist(),
                    dst_excluded_window_counts=excluded_counts,
                    source_sha256=None if source_sha256 is None else str(source_sha256).lower())
    result["edge_index"] = _edge_index(graph)
    result[METADATA_KEY] = np.asarray(json.dumps(metadata, sort_keys=True))
    if savepath is not None:
        path = Path(savepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **result)
    return result


def load_cached_madrid_shift_inputs(path, args):
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = cache_metadata(archive)
        expected = dict(adapter=ADAPTER, cache_version=CACHE_VERSION, zero_is_valid=True,
                        normalize_x=_bool(getattr(args,"normalize_x",False)),
                        normalization_scope=str(getattr(args,"normalization_scope","train")).lower(),
                        x_len=int(getattr(args,"x_len",12)), y_len=int(getattr(args,"y_len",12)),
                        steps_per_day=96, target_scale="source", val_test_mix=False,dst_policy=DST_POLICY)
        mismatch = {k:(metadata.get(k),v) for k,v in expected.items() if metadata.get(k)!=v}
        if mismatch or metadata.get("sample_limits") is not None:
            raise ValueError(f"Madrid cache settings mismatch or partial cache: {mismatch}")
        if _bool(getattr(args,"val_test_mix",False)):
            raise ValueError("Madrid chronological cache cannot use val_test_mix")
        rates = [float(getattr(args,k,v)) for k,v in (("train_rate",.6),("val_rate",.2),("test_rate",.2))]
        if not np.allclose(metadata.get("rates",[]),rates,atol=1e-12,rtol=0):
            raise ValueError("Madrid cache split rates do not match")
        source_steps=int(metadata["source_steps"])
        boundaries=(0,int(source_steps*rates[0]),int(source_steps*(rates[0]+rates[1])),source_steps)
        expected_offsets=dict(zip(SPLITS,boundaries[:-1]))
        expected_steps={split:end-start for split,start,end in zip(SPLITS,boundaries,boundaries[1:])}
        if metadata.get("split_offsets")!=expected_offsets or metadata.get("split_steps")!=expected_steps:
            raise ValueError("Madrid cache split metadata does not match chronological source boundaries")
        for field,key in (("graph_size","nodes"),("expected_source_steps","source_steps")):
            if hasattr(args,field) and int(getattr(args,field))!=int(metadata[key]):
                raise ValueError(f"Madrid cache {key} does not match current source")
        if getattr(args,"source_sha256",None) is not None:
            if metadata.get("source_sha256")!=str(args.source_sha256).lower():
                raise ValueError("Madrid cache source SHA256 does not match the prepared source file")
        result = {key:archive[key].copy() for key in archive.files}
    nodes = int(metadata["nodes"])
    validate_node_ids(result.get("node_ids"),nodes)
    if getattr(args,"expected_node_ids",None) is not None:
        expected_ids=validate_node_ids(args.expected_node_ids,nodes)
        if not np.array_equal(result["node_ids"],expected_ids):
            raise ValueError("Madrid cache node ID order does not match the prepared source")
    _mask(result.get("active_nodes"),(nodes,),"active_nodes")
    first_minute=int(np.datetime64(metadata["first_timestamp"],"m").astype(np.int64))
    clock=np.datetime64(metadata["first_timestamp"],"ns")+np.arange(source_steps)*np.timedelta64(15,"m")
    dst=dst_clock_mask(clock)
    if not np.array_equal(_mask(result.get("dst_invalid_clock"),(source_steps,),"dst_invalid_clock"),dst):
        raise ValueError("Madrid cache DST clock mask differs from actual source calendar")
    if metadata.get("dst_invalid_clock_indices")!=np.flatnonzero(dst).tolist() or metadata.get("last_timestamp")!=str(clock[-1]):
        raise ValueError("Madrid cache DST/calendar metadata is inconsistent")
    for name,field in (("expected_source_first_timestamp","first_timestamp"),("expected_source_last_timestamp","last_timestamp")):
        if hasattr(args,name) and str(getattr(args,name))!=metadata[field]:
            raise ValueError("Madrid cache timestamps differ from the prepared source")
    for split in SPLITS:
        x,y = result.get(f"{split}_x"),result.get(f"{split}_y")
        if x is None or y is None or x.dtype != np.float32 or y.dtype != np.float32 or x.ndim!=3 or y.ndim!=3:
            raise ValueError(f"Madrid cache {split} requires float32 windows")
        if x.shape!=(len(y),expected["x_len"],nodes) or y.shape!=(len(x),expected["y_len"],nodes):
            raise ValueError(f"Madrid cache {split} window shapes are invalid")
        expected_count=expected_steps[split]-expected["x_len"]-expected["y_len"]
        if expected_count<=0 or len(x)!=expected_count:
            raise ValueError(f"Madrid cache {split} must contain every formal within-split window")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Madrid cache {split} contains nonfinite model input/target placeholders")
        for kind,length,array in (("x",expected["x_len"],x),("y",expected["y_len"],y)):
            _mask(result.get(f"{split}_{kind}_valid"),array.shape,f"{split}_{kind}_valid")
            source_index=result.get(f"{split}_{kind}_index")
            if (source_index is None or source_index.shape!=(len(x),length)
                    or source_index.dtype.kind not in "iu" or (source_index<0).any()
                    or (source_index>=metadata["source_steps"]).any()
                    or (length>1 and not np.all(np.diff(source_index,axis=1)==1))):
                raise ValueError(f"Madrid cache {split}_{kind}_index does not map consecutive source rows")
            starts=expected_offsets[split]+np.arange(expected_count-1,-1,-1,dtype=np.int64)
            if kind=="y":
                starts=starts+expected["x_len"]
            canonical=starts[:,None]+np.arange(length,dtype=np.int64)
            if not np.array_equal(source_index,canonical):
                raise ValueError(f"Madrid cache {split}_{kind}_index violates newest-first within-split windows")
            minutes=first_minute+15*source_index
            for suffix,maximum in (("tod",96),("dow",7)):
                indices=result.get(f"{split}_{kind}_{suffix}")
                if indices is None or indices.shape!=(len(x),length) or indices.dtype.kind not in "iu" or (indices<0).any() or (indices>=maximum).any():
                    raise ValueError(f"Madrid cache {split}_{kind}_{suffix} indices are invalid")
                expected_clock=(minutes%1440)//15 if suffix=="tod" else (minutes//1440+3)%7
                if not np.array_equal(indices,expected_clock):
                    raise ValueError(f"Madrid cache {split}_{kind}_{suffix} disagrees with real source calendar")
        if not np.all(result[f"{split}_y_index"][:,0]==result[f"{split}_x_index"][:,-1]+1):
            raise ValueError(f"Madrid cache {split} target does not immediately follow its history")
        x_index,y_index=result[f"{split}_x_index"],result[f"{split}_y_index"]
        excluded=dst[x_index].any(1)|dst[y_index].any(1)
        stored=_mask(result.get(f"{split}_dst_excluded"),(len(x_index),),f"{split}_dst_excluded")
        if not np.array_equal(stored,excluded) or metadata.get("dst_excluded_window_counts",{}).get(split)!=int(excluded.sum()):
            raise ValueError("Madrid cache DST whole-window exclusion metadata differs")
        if result[f"{split}_y_valid"][excluded].any() or result[f"{split}_y"][excluded].any():
            raise ValueError("Madrid cache must exclude complete targets of DST-touching samples")
        if result[f"{split}_x_valid"][dst[x_index]].any():
            raise ValueError("Madrid cache history cannot mark DST clock slots observed")
    return result


class MadridShiftSpatioTemporalDataset(Dataset):
    """PyG nodes x time tensors with source masks and real calendar features."""
    def __init__(self, inputs, split, steps_per_day=96):
        if split not in SPLITS or int(steps_per_day)!=96:
            raise ValueError("Madrid requires train/val/test and steps_per_day=96")
        self.inputs,self.split=inputs,split
        self.x,self.y=inputs[f"{split}_x"],inputs[f"{split}_y"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        data=Data(x=torch.as_tensor(self.x[index].T,dtype=torch.float32),
                  y=torch.as_tensor(self.y[index].T,dtype=torch.float32),
                  x_valid=torch.as_tensor(self.inputs[f"{self.split}_x_valid"][index].T,dtype=torch.bool),
                  y_valid=torch.as_tensor(self.inputs[f"{self.split}_y_valid"][index].T,dtype=torch.bool))
        for kind in ("x","y"):
            for clock in ("tod","dow"):
                setattr(data,f"{kind}_{clock}",torch.as_tensor(self.inputs[f"{self.split}_{kind}_{clock}"][index],dtype=torch.long))
        data.node_ids=torch.as_tensor(self.inputs["node_ids"],dtype=torch.long)
        data.active_nodes=torch.as_tensor(self.inputs["active_nodes"],dtype=torch.bool)
        return data


SpatioTemporalDataset = MadridShiftSpatioTemporalDataset
