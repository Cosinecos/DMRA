"""Numeric checkpoint selection and strict reconstruction of cumulative LoRA."""

from __future__ import annotations

import copy
from dmra.compat import canonical_metadata
import logging
import math
from pathlib import Path
import tempfile

import torch


METADATA_VERSION = 1
MODEL_CLASS = "ForecastBackbone"
LOSS_FLAGS = ("use_mae_loss", "use_freq_loss", "use_bank_ortho_loss",
              "use_node_stability_loss", "use_distillation_loss")


def _bool(value):
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "yes", "y", "on"}:
            return True
        if value.strip().lower() in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")
    return bool(value)


def _switches(args):
    result = {"normalize_x": _bool(getattr(args, "normalize_x", False)),
              "use_residual_norm": True, "use_lora": True, "use_gcn_adapter_lora": True,
              "normalization_scope": str(getattr(args, "normalization_scope", "split")).lower(),
              "init": True,
              "learnable_node_graph_scale": _bool(getattr(args, "learnable_node_graph_scale", True))}
    for name in LOSS_FLAGS:
        result[name] = _bool(getattr(args, name, True))
    return result


def _architecture(args):
    hidden = int(args.model["hidden_channel"])
    return {
        "hidden_channel": hidden, "x_len": int(args.x_len), "y_len": int(args.y_len),
        "dropout": float(getattr(args, "dropout", 0.0)),
        "steps_per_day": int(getattr(args, "steps_per_day", 288)),
        "season_rank": int(getattr(args, "season_rank", max(8, hidden // 8))),
        "gate_rank": int(getattr(args, "gate_rank", max(8, hidden // 8))),
        "res_norm_eps": float(getattr(args, "res_norm_eps", 1e-5)),
        "node_graph_topk": int(getattr(args, "node_graph_topk", getattr(args, "proto_graph_topk", 32))),
        "standard_gcn_tcn_kernel_size": int(getattr(args, "standard_gcn_tcn_kernel_size", 3)),
        "standard_gcn_tcn_dilation": int(getattr(args, "standard_gcn_tcn_dilation", 1)),
        "lora_rank": int(getattr(args, "lora_rank", 4)),
        "lora_mlp_layers": int(getattr(args, "lora_mlp_layers", 3)),
        "trend_bank_size": int(getattr(args, "trend_bank_size", 64)),
        "trend_future_layers": int(getattr(args, "trend_future_layers", 3)),
    }


def _best_checkpoint(directory):
    directory = Path(directory)
    candidates = []
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".pkl", ".pt", ".pth"}:
            continue
        try:
            value = float(path.stem)
        except ValueError:
            continue
        if math.isfinite(value):
            candidates.append((value, path.name, path))
    if not candidates:
        raise FileNotFoundError(f"No numeric checkpoints found in {directory}")
    _, _, path = min(candidates)
    return path, path.stem


def _checkpoint_path(args):
    explicit = getattr(args, "checkpoint_path", "")
    use_pretrained = ((_bool(getattr(args, "load_first_year", False))
                       and int(args.year) <= int(args.begin_year) + 1)
                      or not _bool(getattr(args, "train", True)))
    if not explicit and not use_pretrained:
        return _best_checkpoint(Path(args.path) / str(int(args.year) - 1))
    configured = explicit or getattr(args, "first_year_model_path", "") or getattr(args, "pretrained_path", "")
    if not configured:
        raise ValueError("Set first_year_model_path or checkpoint_path for pretrained/evaluation loading")
    path = Path(configured)
    if path.is_dir():
        year = args.begin_year if _bool(getattr(args, "load_first_year", False)) else args.year
        child = path / str(year)
        return _best_checkpoint(child if child.is_dir() else path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path, path.stem


def _restore_lora_slots(model, state_dict):
    # The six MLP layers have bias deltas; Adapter layers do not.
    return model.restore_lora_slots(state_dict)


def _validate_metadata(metadata, args):
    metadata = canonical_metadata(metadata)
    if metadata.get("format_version") != METADATA_VERSION:
        raise ValueError("Unsupported checkpoint metadata version")
    if int(metadata["begin_year"]) != int(args.begin_year):
        raise ValueError("Checkpoint begin_year does not match this task sequence")
    if getattr(args, "method", None) is not None and metadata.get("method") != args.method:
        raise ValueError("Checkpoint method does not match the requested model")
    saved = dict(metadata["switches"])
    # Checkpoints written before this control always enabled annual adapters.
    saved.setdefault("use_lora", True)
    current = _switches(args)
    missing = set(current) - set(saved)
    if missing:
        raise ValueError(f"Checkpoint switch metadata is incomplete: {sorted(missing)}")
    mismatches = {key: (saved.get(key, False), current.get(key, False))
                  for key in set(saved) | set(current)
                  if saved.get(key, False) != current.get(key, False)}
    expected_arch = _architecture(args)
    for key, value in expected_arch.items():
        if metadata["architecture"].get(key) != value:
            mismatches[key] = (metadata["architecture"].get(key), value)
    if mismatches:
        raise ValueError(f"Checkpoint settings mismatch (saved, requested): {mismatches}")


def _constructor(args):
    methods = getattr(args, "methods", {})
    if getattr(args, "method", None) in methods:
        return methods[args.method]
    from dmra.madrid_model import model_class
    return model_class(args)


def load_best_model(args):
    """Return (strictly restored model, numeric loss string or explicit stem)."""
    path, loss = _checkpoint_path(args)
    logger = getattr(args, "logger", logging.getLogger(__name__))
    logger.info("[*] load from %s", path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dictionary or contain model_state_dict")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None and checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict")
    metadata = checkpoint.get("metadata")
    if metadata is None:
        logger.info("Checkpoint metadata unavailable: legacy weights load strictly, but configuration switches cannot be verified.")
    else:
        _validate_metadata(metadata, args)
    model_args = copy.copy(args)
    for name in ("node_embedding", "assignment_logits", "pattern_bank"):
        if name in state:
            model_args.base_node_size = int(state[name].shape[0])
            model_args.graph_size = model_args.base_node_size
            break
    if metadata is not None:
        model_args.year = int(metadata["saved_year"])
    model = _constructor(args)(model_args)
    years = _restore_lora_slots(model, state)
    if metadata is not None:
        structure = metadata["structure"]
        if structure["num_nodes"] != int(model.num_nodes) or structure["lora_years"] != years:
            raise ValueError("Checkpoint structural metadata does not match its saved tensor keys")
    model.load_state_dict(state, strict=True)
    model.args = args
    return model.to(getattr(args, "device", "cpu")), loss


def save_checkpoint(model, args, path):
    """Save weights plus architecture/switch metadata; keep the legacy state key."""
    state = model.state_dict()
    years = model.saved_lora_years()
    metadata = {
        "format_version": METADATA_VERSION, "model_class": type(model).__name__,
        "method": getattr(args, "method", MODEL_CLASS), "begin_year": int(args.begin_year),
        "saved_year": int(args.year), "architecture": _architecture(args),
        "switches": _switches(args),
        "structure": {"num_nodes": int(model.num_nodes), "lora_years": years},
        "loss_weights": {name: float(getattr(args, name, default)) for name, default in
                         (("lambda_mae", 1.0), ("lambda_freq", 0.0), ("lambda_bank_ortho", 0.0),
                          ("lambda_u", 0.0), ("lambda_dist", 0.0))},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        torch.save({"model_state_dict": state, "metadata": metadata}, temporary)
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata
