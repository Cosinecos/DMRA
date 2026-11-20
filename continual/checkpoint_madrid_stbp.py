"""Strict, dataset-bound STBP best weights for the selected Madrid chain."""
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import random

import numpy as np
import torch

from checkpoint_madrid_shift import protocol as madrid_protocol
from model_madrid_stbp import METHOD, MadridSTBP, registry_identity


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol(args):
    result = madrid_protocol(args)
    result.update(method=METHOD, model=copy.deepcopy(args.model), x_len=int(args.x_len), y_len=int(args.y_len),
                  dropout=float(args.dropout), normalize_x=bool(args.normalize_x),
                  normalization_scope=args.normalization_scope if args.normalize_x else None,
                  rates=[float(getattr(args,k,v)) for k,v in (('train_rate',.6),('val_rate',.2),('test_rate',.2))],
                  window_mode="smoke_8_4_4" if getattr(args,'smoke',False) else "full_windows",
                  init=bool(args.init), begin_year=int(args.begin_year),
                  training="first_all_then_pattern_bank_only" if args.init else "independent_task_retraining",
                  availability="active_subset_attention_and_norm_inactive_self_only",
                  physical_edges="STBP_predictor_uses_node_count_not_edge_weights")
    return result


@contextmanager
def _preserve_rng_state():
    python_state=random.getstate();numpy_state=np.random.get_state();cpu_state=torch.get_rng_state()
    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state);torch.set_rng_state(cpu_state)
        if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(model, args, path, val_mae):
    if not isinstance(model, MadridSTBP) or args.method != METHOD:
        raise ValueError("Madrid STBP saver requires its independent model/method")
    month = int(args.selected_months[int(args.year)])
    metadata = dict(format_version=1, method=METHOD, protocol=protocol(args), saved_year=int(args.year),
                    task_month=month, task_label=f"{args.year}-{month:02d}", source_sha256=args.source_sha256,
                    graph_sha256=sha256(Path(args.graph_path)/f"{args.year}_adj.npz"),
                    node_identity=registry_identity(model.state_dict()), val_mae=float(val_mae))
    value = dict(metadata=metadata, model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()})
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as f:
            temporary = Path(f.name)
        torch.save(value,temporary); os.replace(temporary,path); temporary=None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata


def load_checkpoint(path, args, expected_year=None):
    if args.method != METHOD:
        raise ValueError("Madrid STBP loading requires its independent method")
    value = torch.load(path,map_location="cpu",weights_only=False)
    metadata = value.get("metadata",{})
    if metadata.get("format_version") != 1 or metadata.get("method") != METHOD or metadata.get("protocol") != protocol(args):
        raise ValueError("Madrid STBP checkpoint model/data/preprocessing protocol mismatch")
    saved_year = int(metadata.get("saved_year",-1))
    if expected_year is not None and saved_year != int(expected_year):
        raise ValueError("Madrid STBP checkpoint belongs to a different task year")
    month = args.selected_months.get(saved_year,args.selected_months.get(str(saved_year)))
    if month is None or metadata.get("task_month") != int(month) or metadata.get("task_label") != f"{saved_year}-{int(month):02d}":
        raise ValueError("Madrid STBP checkpoint selected month differs")
    if metadata.get("source_sha256") != sha256(Path(args.raw_data_path)/f"{saved_year}.npz"):
        raise ValueError("Madrid STBP checkpoint source changed")
    if metadata.get("graph_sha256") != sha256(Path(args.graph_path)/f"{saved_year}_adj.npz"):
        raise ValueError("Madrid STBP checkpoint graph changed")
    state = value.get("model_state_dict",{})
    identity = registry_identity(state)
    if identity != metadata.get("node_identity"):
        raise ValueError("Madrid STBP checkpoint node identity metadata mismatch")
    ids = state['registry_node_ids'].tolist()
    expected = getattr(args,'expected_node_ids',None)
    if expected is not None and ids != list(expected)[:len(ids)]:
        raise ValueError("Madrid STBP checkpoint node IDs do not match expected stable prefix")
    saved_args = copy.copy(args)
    saved_args.year = saved_year; saved_args.base_node_size = identity['nodes']
    with _preserve_rng_state():
        model = MadridSTBP(saved_args)
        model.load_state_dict(state,strict=True)
        model = model.to(args.device); model.args = args
    return model,float(metadata['val_mae'])
