"""EAC/PECPM/STKEC/TrafficStream on full Madrid chronological task chains."""
import argparse
from copy import deepcopy
from datetime import datetime
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from data_madrid_shift import _forward_fill, calendar_indices, ensure_node_prefix, file_hash, load_madrid_stage
from madrid_continual_runtime.baselines import canonical_model, implementation
from madrid_continual_runtime.common import fit_task


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ("EAC", "PECPM", "STKEC", "TS")
DATASET = "MADRID_2022_2026_MMD"
PROTOCOL = "psrag_madrid_full_continual_60_20_20_v1"


class WindowView:
    """Keep the full window tensor as a view; copy only selected slices.

    Native TrafficStream converts batches of newest-first windows directly to
    torch.as_tensor, which cannot consume NumPy negative strides. A selected
    slice therefore receives a positive-stride copy, while constructing the
    full train/validation/test inputs never materializes all overlapping X/Y.
    """
    def __init__(self, array):
        self.array = array

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def ndim(self):
        return self.array.ndim

    def __len__(self):
        return len(self.array)

    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.array, dtype=dtype)
        return result.copy() if copy else result

    def __getitem__(self, key):
        result = self.array[key]
        return result.copy() if isinstance(result, np.ndarray) and any(stride < 0 for stride in result.strides) else result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="all", help="EAC/PECPM/STKEC/TS, comma-separated names, or all")
    p.add_argument("--gpuid", type=int, default=0, help="-1 explicitly selects CPU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", "--epoch", dest="epoch", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=.005)
    p.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs_madrid_continual_baselines")
    p.add_argument("--conf", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    p.add_argument("--check-data", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Real nodes, 4/2/2 windows, at most2 epochs/stage")
    p.add_argument("--max-tasks", type=int)
    return p


def merged(left, right):
    result = deepcopy(left)
    for name, value in right.items():
        result[name] = merged(result[name], value) if isinstance(value, dict) and isinstance(result.get(name), dict) else deepcopy(value)
    return result


def configurations(cli):
    supplied = json.loads((HERE / "conf" / "forecast" / "MADRID.json").read_text(encoding="utf-8-sig"))
    models = MODELS if cli.model.lower() == "all" else tuple(canonical_model(name.strip()) for name in cli.model.split(","))
    if not models or len(set(models)) != len(models) or any(model not in MODELS for model in models):
        raise ValueError("Choose distinct EAC/PECPM/STKEC/TS names, or all")
    result = []
    for model in models:
        config = dict(supplied, baseline=model, method=model, dataset=DATASET, variant="raw",
                      seed=cli.seed, gpuid=cli.gpuid, num_workers=cli.num_workers,
                      normalize_x=False, normalization_scope="train",
                      target_scale="source", train_rate=.6, val_rate=.2, test_rate=.2,
                      val_test_mix=False, x_len=12, y_len=12, save_data_path="cache/MADRID_2022_2026_MMD/",
                      dropout=0., weight_decay=.01, grad_clip=0., init=True, train=True,
                      lr_schedule=None, batch_size_schedule=None, cache_data=False, data_process=True,
                      smoke=cli.smoke, load_first_year=False,
                      model={"in_channel": 12, "out_channel": 12, "hidden_channel": 192})
        module = implementation(model)
        config = merged(config, getattr(module, "DEFAULTS", {}))
        if hasattr(module, "defaults"):
            config = merged(config, module.defaults(DATASET))
        if model != "EAC":
            config.pop("model", None)
        # Explicit current-comparison optimization profile, retaining native
        # network widths, objectives, node selection and Fisher algorithms.
        config.update(epoch=cli.epoch, patience=cli.patience, batch_size=cli.batch_size,
                      lr=cli.lr, lr_schedule=None, batch_size_schedule=None)
        if cli.conf:
            config = merged(config, json.loads(cli.conf.read_text(encoding="utf-8-sig")))
        for setting in cli.set:
            if "=" not in setting:
                raise ValueError("--set requires KEY=JSON")
            key, value = setting.split("=", 1)
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            destination = config
            parts = key.split(".")
            for part in parts[:-1]:
                if not isinstance(destination.get(part), dict):
                    raise ValueError("Unknown setting: " + key)
                destination = destination[part]
            if parts[-1] not in destination:
                raise ValueError("Unknown setting: " + key)
            destination[parts[-1]] = value
        fixed = dict(baseline=model, method=model, dataset=DATASET, variant="raw",
                     normalize_x=False, target_scale="source", init=True,
                     train=True, train_rate=.6, val_rate=.2, test_rate=.2,
                     val_test_mix=False, x_len=12, y_len=12, load_first_year=False,
                     cache_data=False, data_process=True)
        for key, expected in fixed.items():
            if config.get(key) != expected:
                raise ValueError("Full Madrid continual protocol fixes " + key)
        for key in ("begin_year", "end_year", "raw_data_path", "graph_path", "steps_per_day"):
            if config.get(key) != supplied[key]:
                raise ValueError("Frozen Madrid source/calendar fixes " + key)
        if config["lr_schedule"] is not None or config["batch_size_schedule"] is not None:
            raise ValueError("This comparison uses explicit fixed lr and batch size")
        if config["normalization_scope"] not in {"train", "split"}:
            raise ValueError("normalization_scope must be train or split")
        if cli.max_tasks is not None and cli.max_tasks <= 0:
            raise ValueError("--max-tasks must be positive")
        if cli.smoke:
            config.update(epoch=min(int(config["epoch"]), 2), batch_size=2, num_workers=0)
        if min(config["epoch"], config["batch_size"], config["lr"]) <= 0 or config["patience"] < 0 or config["num_workers"] < 0:
            raise ValueError("Invalid optimization settings")
        config.update(output_root=str(cli.output_dir.resolve()), max_tasks=cli.max_tasks,
                      training_fraction_of_entire_task=.6,
                      training_profile="full_current_comparison_200ep_pat30",
                      protocol=dict(name=PROTOCOL, split="chronological_per_task", rates=[.6, .2, .2],
                                    training_fraction_of_entire_task=.6,
                                    strategy="native_continual_weight_inheritance",
                                    model_reinitialized_each_stage=False,
                                    validation="global_observed_MAE", test="global_observed_prefix_metrics",
                                    targets="source", input_preprocessing="source_x",
                                    normalization_scope=config["normalization_scope"],
                                    method_training_loss="upstream_specific", old_task_backtesting=False,
                                    storage="zero_copy_sliding_window_views_no_window_cache"))
        result.append(config)
    return result


def _windows(values, count):
    """Newest-first sample views over [time, node]."""
    return np.lib.stride_tricks.sliding_window_view(values, 24, axis=0).transpose(0, 2, 1)[:count][::-1]


def load_context(config, year):
    raw_path = ROOT / config['raw_data_path'] / f'{year}.npz'
    graph_path = ROOT / config['graph_path'] / f'{year}_adj.npz'
    source, graph, label = load_madrid_stage(raw_path, graph_path, year, config['train_rate'])
    raw, valid, dst = source['x'], source['valid'], source['dst_invalid_clock']
    node_ids, active = source['node_ids'], source['active_nodes']
    tod, dow = calendar_indices(source['timestamps'])
    stop = int(len(raw) * config['train_rate'])
    val_stop = int(len(raw) * (config['train_rate'] + config['val_rate']))
    bounds = (0, stop, val_stop, len(raw))
    filled, _ = _forward_fill(raw, valid, stop)
    inputs = {"node_ids": node_ids, "active_nodes": active}
    sample_counts, split_steps, dst_counts = {}, {}, {}
    for split, lo, hi in zip(("train", "val", "test"), bounds, bounds[1:]):
        count = hi - lo - 24
        if config['smoke']:
            count = min(count, {'train': 4, 'val': 2, 'test': 2}[split])
        if count <= 0:
            raise ValueError(f'{split} has insufficient steps for the common 12-to-12 setup in {year}')
        history = filled[lo:hi]
        truth = np.where(valid[lo:hi], raw[lo:hi], 0.).astype(np.float32)
        windows = _windows(history, count)
        observed = _windows(valid[lo:hi], count)
        target_y = _windows(truth, count)[:, 12:]
        target_valid = observed[:, 12:]
        dst_invalid = np.lib.stride_tricks.sliding_window_view(dst[lo:hi], 24)[:count][::-1].any(1)
        if dst_invalid.any():
            target_valid = target_valid.copy()
            target_valid[dst_invalid] = False
            target_y = target_y.copy()
            target_y[dst_invalid] = 0.
        starts = lo + np.arange(count - 1, -1, -1, dtype=np.int64)
        x_index = starts[:, None] + np.arange(12)
        y_index = starts[:, None] + np.arange(12, 24)
        inputs.update({split + "_x": WindowView(windows[:, :12]),
                       split + "_y": WindowView(target_y),
                       split + "_x_valid": WindowView(observed[:, :12]), split + "_y_valid": WindowView(target_valid),
                       split + "_x_index": x_index, split + "_y_index": y_index,
                       split + "_x_tod": tod[x_index], split + "_x_dow": dow[x_index],
                       split + "_y_tod": tod[y_index], split + "_y_dow": dow[y_index]})
        sample_counts[split], split_steps[split], dst_counts[split] = count, hi - lo, int(dst_invalid.sum())
    metadata = dict(protocol=PROTOCOL, dataset=DATASET, year=year, task_label=label, nodes=len(node_ids),
                    train_allowed_stop=stop, training_fraction_of_entire_task=.6,
                    source_steps=len(raw), samples=sample_counts, split_steps=split_steps,
                    dst_excluded_windows=dst_counts, normalize_x=False, normalization_scope=config['normalization_scope'],
                    storage="zero_copy_sliding_window_views_no_window_cache",
                    source_sha256=file_hash(raw_path), graph_source_sha256=file_hash(graph_path),
                    graph_sha256=hashlib.sha256(graph.tobytes()).hexdigest())
    semantic = {key: metadata[key] for key in ("protocol", "source_sha256", "graph_sha256", "year", "train_allowed_stop", "normalize_x", "normalization_scope")}
    plan_sha256 = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    metadata["plan_sha256"] = plan_sha256
    inputs["__metadata__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return SimpleNamespace(inputs=inputs, raw_adj=graph, node_ids=node_ids,
                           active_nodes=active, metadata=metadata, preprocessing=metadata,
                           plan_sha256=plan_sha256, train_allowed_stop=stop, train_raw_data=filled[:stop],
                           train_raw_valid=valid[:stop], year=year, dataset=DATASET)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(config):
    args = SimpleNamespace(**deepcopy(config))
    args.config_identity = deepcopy(config)
    if args.gpuid < 0:
        args.device = torch.device("cpu")
    elif torch.cuda.is_available() and args.gpuid < torch.cuda.device_count():
        args.device = torch.device(f"cuda:{args.gpuid}")
    else:
        raise RuntimeError("Requested CUDA unavailable; explicitly use --gpuid -1 for CPU")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.path = str(Path(args.output_root) / args.baseline / DATASET / args.variant / f"{stamp}_seed{args.seed}_{uuid.uuid4().hex[:6]}")
    Path(args.path).mkdir(parents=True)
    args.logger = logging.getLogger(args.path); args.logger.setLevel(logging.INFO); args.logger.propagate = False
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(args.path) / "train.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")); args.logger.addHandler(handler)
    write_json(Path(args.path) / "config.json", config)
    args.logger.info("full Madrid baseline=%s variant=%s device=%s settings=%s", args.baseline, args.variant, args.device, config)
    adapter = implementation(args.baseline).Adapter(args)
    previous = None; previous_adj = None; results = {}
    years = list(range(args.begin_year, args.end_year + 1))
    if args.max_tasks is not None:
        years = years[:args.max_tasks]
    try:
        for year in years:
            args.year = year
            data = load_context(config, year)
            if previous is not None:
                ensure_node_prefix(previous.node_ids, data.node_ids)
            raw = np.asarray(data.raw_adj, dtype=np.float32)
            adjacency = torch.as_tensor(raw / (raw.sum(1, keepdims=True) + 1e-6), dtype=torch.float32, device=args.device)
            args.graph_size = len(raw); args.adj = adjacency; args.sub_adj = adjacency; args.prev_adj = previous_adj
            ctx = SimpleNamespace(**vars(data), args=args, logger=args.logger, adj=adjacency,
                                  raw_data=data.train_raw_data,
                                  previous_raw_data=None if previous is None else previous.train_raw_data,
                                  previous_raw_valid=None if previous is None else previous.train_raw_valid,
                                  previous_inputs=None if previous is None else previous.inputs,
                                  previous_raw_adj=None if previous is None else previous.raw_adj)
            args.logger.info("task=%s nodes=%s full_training_rows=%s preprocessing=%s", year, len(raw), data.train_allowed_stop, data.metadata)
            results[str(year)] = fit_task(adapter, ctx)
            write_json(Path(args.path) / "results.json", results)
            previous, previous_adj = data, adjacency
        average = {h: {metric: float(np.mean([task["metrics"][h][metric] for task in results.values() if task["metrics"][h][metric] is not None]))
                       if any(task["metrics"][h][metric] is not None for task in results.values()) else None
                       for metric in ("MAE", "RMSE", "MAPE")} for h in next(iter(results.values()))["metrics"]}
        summary = dict(model=args.baseline, dataset=DATASET, variant=args.variant, run=args.path,
                       smoke=args.smoke, tasks=list(results), average=average, protocol=config["protocol"])
        write_json(Path(args.path) / "summary.json", summary); args.logger.info("COMPLETED %s", summary)
        return summary
    finally:
        for handler in list(args.logger.handlers):
            handler.close(); args.logger.removeHandler(handler)
        del adapter; gc.collect()
        if args.device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    cli = parser().parse_args()
    configs = configurations(cli)
    if cli.plan:
        print(json.dumps(configs, ensure_ascii=False, indent=2)); return
    if cli.check_data:
        checks = []
        config = configs[0]
        previous_ids = None
        for year in range(config["begin_year"], config["end_year"] + 1):
            ctx = load_context(config, year)
            if previous_ids is not None:
                ensure_node_prefix(previous_ids, ctx.node_ids)
            previous_ids = ctx.node_ids
            checks.append(dict(year=year, nodes=len(ctx.node_ids), train_allowed_stop=ctx.train_allowed_stop,
                               source_steps=ctx.metadata["source_steps"], protocol=PROTOCOL,
                               samples={s: len(ctx.inputs[s + "_x"]) for s in ("train", "val", "test")},
                               source_sha256=ctx.metadata["source_sha256"], plan_sha256=ctx.plan_sha256))
        print(json.dumps(dict(status="passed", checks=checks), ensure_ascii=False, indent=2)); return
    for config in configs:
        run(config)


if __name__ == "__main__":
    main()
