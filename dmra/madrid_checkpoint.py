"""Strict method-, embedding-, source- and registry-bound Madrid checkpoints."""
import os
import hashlib
from contextlib import contextmanager
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from dmra.runtime.madrid_core import checkpoint as core
from dmra.compat import canonical_metadata
from dmra.model import embedding_architecture
from dmra.madrid_model import model_class, method_name, variant_name


def registry_identity(state):
    embedding=state.get("node_embedding")
    ids=state.get("registry_node_ids")
    active=state.get("task_graph_active")
    if not torch.is_tensor(embedding) or not torch.is_tensor(ids) or not torch.is_tensor(active):
        raise ValueError("Madrid checkpoint is missing its stable node registry")
    nodes=embedding.shape[0]
    if ids.shape!=(nodes,) or active.shape!=(nodes,) or ids.dtype!=torch.int64 or active.dtype!=torch.bool:
        raise ValueError("Madrid checkpoint registry types/shapes are invalid")
    if (ids<0).any() or ids.unique().numel()!=nodes:
        raise ValueError("Madrid checkpoint node IDs must be unique and nonnegative")
    return dict(nodes=nodes,node_ids_sha256=hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest(),
                graph_active_sha256=hashlib.sha256(active.detach().cpu().numpy().tobytes()).hexdigest())


def embedding_settings(args):
    value = embedding_architecture(args)
    return value


def protocol(args):
    return dict(version=1, dataset="MADRID_2022_2026_MMD",
                data_manifest_sha256=str(args.data_manifest_sha256),
                selection_result_sha256=str(args.selection_result_sha256),
                reference_anchor_sha256=str(args.reference_anchor_sha256),
                rates=[float(getattr(args, name)) for name in ("train_rate", "val_rate", "test_rate")],
                window_mode="smoke_8_4_4" if getattr(args, "smoke", False) else "full_windows",
                calendar="actual_madrid_selected_month_wall_clock_15min",
                missing_labels="explicit_mask_zero_is_valid_and_DST_window_exclusion",
                validation="global_observed_MAE", unit="vehicles_per_hour",
                mape="observed_positive_targets_only", model=model_class(args).__name__,
                embedding=embedding_settings(args))


@contextmanager
def _preserve_rng_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _read_checked(path, args):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected an independent Madrid GCN-Adapter checkpoint")
    meta = canonical_metadata(payload.get("metadata", {}))
    if meta.get("method") != method_name(variant_name(args)) or meta.get("model_class") != model_class(args).__name__:
        raise ValueError("Checkpoint is not the requested independent Madrid GCN-Adapter model")
    if meta.get("madrid_protocol") != protocol(args):
        raise ValueError("Madrid GCN-Adapter checkpoint data/protocol differs from the requested run")
    identity = registry_identity(payload.get("model_state_dict", {}))
    if meta.get("madrid_node_identity") != identity:
        raise ValueError("Madrid registry metadata does not match saved tensors")
    for name, value in embedding_settings(args).items():
        if meta.get("architecture", {}).get(name) != value:
            raise ValueError("Madrid GCN-Adapter architecture metadata mismatch: " + name)
    year = int(meta.get("saved_year", -1))
    months = args.selected_months
    month = months.get(year, months.get(str(year)))
    if month is None or meta.get("madrid_shift_task") != dict(year=year, month=int(month), label=f"{year}-{int(month):02d}"):
        raise ValueError("Madrid saved month differs from frozen selection")
    return payload


def load_best_model(args):
    method = method_name(variant_name(args))
    if args.method != method:
        raise ValueError("Madrid GCN-Adapter method differs from its requested variant")
    path, _ = core._checkpoint_path(args)
    _read_checked(path, args)
    args.methods = {method: model_class(args)}
    with _preserve_rng_state():
        model, loss = core.load_best_model(args)
    if type(model) is not model_class(args):
        raise ValueError("Unexpected Madrid GCN-Adapter constructor")
    return model, loss


def save_checkpoint(model, args, path):
    if type(model) is not model_class(args) or args.method != method_name(variant_name(args)):
        raise ValueError("Wrong independent model for Madrid GCN-Adapter checkpoint")
    identity = registry_identity(model.state_dict())
    metadata = core.save_checkpoint(model, args, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata["architecture"].update(embedding_settings(args))
    metadata["madrid_protocol"] = protocol(args)
    metadata["madrid_node_identity"] = identity
    month = int(args.selected_months.get(int(args.year), args.selected_months.get(str(args.year))))
    metadata["madrid_shift_task"] = dict(year=int(args.year), month=month, label=f"{int(args.year)}-{month:02d}")
    payload["metadata"] = metadata
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata
