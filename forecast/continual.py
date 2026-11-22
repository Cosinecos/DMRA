"""Strict, audited transfer of a preceding validation-selected forecast state.

Models are built for the current task first. The four supported adapters
have a small, explicit set of node-indexed weights.  Only those axes may grow;
all other state keys, shapes and dtypes must agree.  In particular, a new
graph is task context and must never be replaced by a checkpoint's old graph.
Native fixed parameters (PatchTST's attention scales) keep their original
``requires_grad`` flags.  Optimizer state is deliberately outside this API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import torch

from . import canonical_model


CHECKPOINT_FORMAT = "dmra-forecast-full-ft-v1"
_TASK_BUFFERS = {
    "physical_forward", "physical_backward", "active_nodes",
    "task_graph_active", "node_ids", "registry_node_ids",
}
_CONFIG_IDENTITY = (
    "steps_per_day", "x_len", "y_len", "train_rate", "val_rate", "test_rate", "pems_days",
    "smoke",
)
_PREPROCESSING_IDENTITY = (
    "normalize_x", "target_scale", "calendar", "mask",
    "window_order", "split_final_candidate", "storage",
)
_PATCH_AFFINE = re.compile(
    r"core\.(?:model|model_trend|model_res)\.revin_layer\."
    r"affine_(?:weight|bias)\Z"
)
_PATCH_HEAD = re.compile(
    r"core\.(?:model|model_trend|model_res)\.head\.linears\."
    r"(?P<node>\d+)\.(?:weight|bias)\Z"
)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Match the common trainer's state hash, without importing that trainer."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
        try:
            raw = value.numpy().tobytes()
        except TypeError:  # NumPy does not expose bfloat16 directly.
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_identity(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _node_ids(value, count, label):
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != count or array.dtype.kind not in "iu":
        raise ValueError(f"{label} node_ids must be an integer vector of length {count}")
    if (array < 0).any() or len(np.unique(array)) != count:
        raise ValueError(f"{label} node_ids must be unique and nonnegative")
    # Conversion through Python integers rejects uint64 overflow as well.
    ids = [int(item) for item in array]
    if any(item > np.iinfo(np.int64).max for item in ids):
        raise ValueError(f"{label} node_ids exceed the supported int64 registry")
    return ids


def _same_field(previous, current, name, where):
    if name not in previous or name not in current:
        raise ValueError(f"Missing {where} identity field: {name}")
    if _json_identity(previous[name]) != _json_identity(current[name]):
        raise ValueError(f"Checkpoint {where} identity mismatch: {name}")


def _validate_identity(metadata, task, config):
    if not isinstance(metadata, Mapping):
        raise ValueError("Forecast checkpoint requires structured metadata")
    fmt = metadata.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported forecast checkpoint format: {fmt!r}")
    if metadata.get("strategy") != "full_ft":
        raise ValueError("Checkpoint format/strategy identity mismatch")
    model_name = canonical_model(config["model_name"])
    if canonical_model(str(metadata.get("model", ""))) != model_name:
        raise ValueError("Checkpoint model identity mismatch")
    for name in ("dataset", "variant"):
        if metadata.get(name) != config.get(name):
            raise ValueError(f"Checkpoint {name} identity mismatch")
    previous_year = metadata.get("year")
    if isinstance(previous_year, bool) or not isinstance(previous_year, (int, np.integer)):
        raise ValueError("Checkpoint year must be an integer")
    if int(previous_year) != int(task.year) - 1:
        raise ValueError("Checkpoint must come from the immediately preceding stage year")
    previous_config = metadata.get("config")
    if not isinstance(previous_config, Mapping):
        raise ValueError("Checkpoint is missing its resolved model/data configuration")
    _same_field(previous_config, config, "model", "configuration")
    for name in ("model_name", "dataset", "variant"):
        _same_field(previous_config, config, name, "configuration")
    for name in _CONFIG_IDENTITY:
        _same_field(previous_config, config, name, "data protocol")
    if int(config["x_len"]) != 12 or int(config["y_len"]) != 12:
        raise ValueError("Forecast inheritance requires the common 12->12 protocol")
    if config["model"].get("seq_len") != 12 or config["model"].get("pred_len") != 12:
        raise ValueError("Model sequence lengths disagree with the common 12->12 protocol")
    previous_preprocessing = metadata.get("preprocessing")
    current_preprocessing = getattr(task, "metadata", None)
    if previous_preprocessing is not None or current_preprocessing is not None:
        if not isinstance(previous_preprocessing, Mapping) or not isinstance(current_preprocessing, Mapping):
            raise ValueError("Checkpoint/current preprocessing metadata must both be present")
        for name in _PREPROCESSING_IDENTITY:
            if name in previous_preprocessing or name in current_preprocessing:
                _same_field(previous_preprocessing, current_preprocessing, name, "preprocessing")
    old_count = metadata.get("num_nodes")
    if isinstance(old_count, bool) or not isinstance(old_count, (int, np.integer)) or old_count <= 0:
        raise ValueError("Checkpoint num_nodes must be a positive integer")
    new_count = int(task.nodes)
    old_count = int(old_count)
    old_ids = _node_ids(metadata.get("node_ids"), old_count, "Checkpoint")
    new_ids = _node_ids(task.node_ids, new_count, "Current")
    if new_count < old_count or new_ids[:old_count] != old_ids:
        raise ValueError("Node registry must preserve the old ID prefix; shrinking/reordering is unsupported")
    return model_name, old_count, new_count, old_ids, new_ids


def _node_axis(model_name, name):
    """Return the only audited (axis, contiguous features per node) mappings."""
    if model_name == "TQNET" and name == "core.temporalQuery":
        return 1, 1
    if model_name == "GWNET" and name in ("core.nodevec1", "core.nodevec2"):
        return (0, 1) if name.endswith("nodevec1") else (1, 1)
    if model_name == "STID" and name == "core.node_emb":
        return 0, 1
    if model_name == "PATCHTST" and _PATCH_AFFINE.fullmatch(name):
        return 0, 1
    return None


def inherit_best(model, task, config, previous_checkpoint, device):
    """Load old learned weights, retaining current graph and fresh new-node rows.

    The caller chooses the previous stage's validation-best state. Validation selection
    is the trainer's responsibility; this function validates its identity and
    state transfer rather than searching directories or accepting partial loads.
    A JSON-serializable report accounts for every source and destination tensor.
    """
    path = Path(previous_checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Previous best checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("Forecast inheritance requires state_dict and metadata")
    metadata, old = payload.get("metadata"), payload["state_dict"]
    name, old_n, new_n, old_ids, new_ids = _validate_identity(metadata, task, config)
    if int(getattr(model, "num_nodes", new_n)) != new_n:
        raise ValueError("The destination model was not constructed for the current node count")
    current = model.state_dict()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    fixed = sorted(key for key, value in parameters.items() if not value.requires_grad)
    # No caller-defined freezing is allowed in full_ft.  The one native fixed
    # family is a mathematical attention scale in the pinned PatchTST core.
    for key in fixed:
        if name != "PATCHTST" or not key.endswith(".self_attn.sdp_attn.scale"):
            raise ValueError(f"Unexpected frozen parameter in full fine-tuning: {key}")
    old_only = set(old) - set(current)
    if old_only:
        raise ValueError(f"Checkpoint contains unsupported state keys: {sorted(old_only)}")
    new_only = sorted(set(current) - set(old))
    for key in new_only:
        head = _PATCH_HEAD.fullmatch(key) if name == "PATCHTST" and config["model"].get("individual") else None
        if head is None or not old_n <= int(head.group("node")) < new_n or key not in parameters:
            raise ValueError(f"Unsupported newly introduced state key: {key}")
    merged = {}
    copied, expanded, kept = [], [], []
    copied_parameters = copied_buffers = 0
    fresh_before = state_sha256(current)
    for key, destination in current.items():
        if not torch.is_tensor(destination) or destination.layout != torch.strided:
            raise ValueError(f"Unsupported destination state tensor: {key}")
        destination = destination.detach().cpu().clone()
        if key in new_only:
            merged[key] = destination
            continue
        source = old[key]
        if not torch.is_tensor(source) or source.layout != torch.strided:
            raise ValueError(f"Checkpoint state is not a dense tensor: {key}")
        source = source.detach().cpu()
        if source.dtype != destination.dtype:
            raise ValueError(f"Checkpoint tensor dtype mismatch: {key}")
        if source.is_floating_point() or source.is_complex():
            if not torch.isfinite(source).all():
                raise ValueError(f"Non-finite checkpoint tensor: {key}")
        if key in _TASK_BUFFERS:
            if key not in buffers or key in parameters:
                raise ValueError(f"Task-context exception cannot hide a learned parameter: {key}")
            expected_old = (old_n, old_n) if key.startswith("physical_") else (old_n,)
            expected_new = (new_n, new_n) if key.startswith("physical_") else (new_n,)
            if tuple(source.shape) != expected_old or tuple(destination.shape) != expected_new:
                raise ValueError(f"Task buffer has unexpected registry dimensions: {key}")
            kept.append(key)
            merged[key] = destination
            continue
        axis_spec = _node_axis(name, key)
        if axis_spec is not None:
            axis, block = axis_spec
            if source.ndim <= axis or destination.ndim != source.ndim:
                raise ValueError(f"Invalid node-indexed tensor rank: {key}")
            expected = list(source.shape)
            expected[axis] = new_n * block
            if source.shape[axis] != old_n * block or list(destination.shape) != expected:
                raise ValueError(f"Unexpected node-axis shape change: {key}")
            index = [slice(None)] * source.ndim
            index[axis] = slice(0, old_n * block)
            destination[tuple(index)] = source
            if old_n != new_n:
                expanded.append({"name": key, "axis": axis, "block_size": block,
                                 "old_shape": list(source.shape), "new_shape": list(destination.shape),
                                 "copied_numel": source.numel(),
                                 "fresh_numel": destination.numel() - source.numel()})
        else:
            if source.shape != destination.shape:
                raise ValueError(f"Unsupported checkpoint tensor shape change: {key}")
            destination.copy_(source)
        merged[key] = destination
        copied.append(key)
        if key in parameters:
            copied_parameters += source.numel()
        elif key in buffers:
            copied_buffers += source.numel()
        else:
            raise ValueError(f"State key is neither a registered parameter nor buffer: {key}")
    expected_hash = state_sha256(merged)
    model.load_state_dict(merged, strict=True)
    model.to(device)
    actual_hash = state_sha256(model.state_dict())
    if actual_hash != expected_hash:
        raise RuntimeError("Strict inherited model differs from the audited merged state")
    if fixed != sorted(key for key, value in model.named_parameters() if not value.requires_grad):
        raise RuntimeError("Loading inheritance unexpectedly changed native trainability")
    source_hash = state_sha256(old)
    migration = {"model": name, "source_state_sha256": source_hash,
                 "fresh_initial_state_sha256": fresh_before, "initial_state_sha256": actual_hash,
                 "old_node_ids": old_ids, "new_node_ids": new_ids,
                 "copied_tensor_names": sorted(copied), "new_tensor_names": new_only,
                 "expanded_tensors": expanded, "kept_task_buffers": sorted(kept)}
    return {
        "source_checkpoint": str(path), "source_checkpoint_sha256": _file_sha256(path),
        "source_year": int(metadata["year"]), "source_format": metadata["format"],
        "source_state_sha256": source_hash,
        "fresh_initial_state_sha256": fresh_before, "initial_state_sha256": actual_hash,
        "loaded_initial_state_sha256": actual_hash,
        "migration_sha256": hashlib.sha256(_json_identity(migration).encode("utf-8")).hexdigest(),
        "node_identity_policy": "stable_append_only_prefix", "old_num_nodes": old_n,
        "new_num_nodes": new_n, "copied_tensors": len(copied),
        "copied_tensor_names": sorted(copied), "expanded_tensors": expanded,
        "new_tensor_names": new_only, "kept_task_buffers": sorted(kept),
        "copied_parameter_numel": copied_parameters, "copied_buffer_numel": copied_buffers,
        "fresh_parameter_numel": sum(value.numel() for value in parameters.values()) - copied_parameters,
        "trainable_parameter_numel": sum(value.numel() for value in parameters.values() if value.requires_grad),
        "fixed_parameter_names": fixed, "intrinsic_fixed_parameters": fixed,
        "strict_state_verified": True, "optimizer_state_inherited": False,
    }
