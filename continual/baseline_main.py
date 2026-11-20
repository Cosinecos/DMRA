"""Run six continual forecasting baselines on the four legacy datasets."""

import argparse
from copy import deepcopy
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import random
from types import SimpleNamespace
import uuid

import networkx as nx
import numpy as np
import torch

from baselines import MODELS, canonical_model, implementation
from baselines.common import fit_task
from data import cache_metadata, check_source_files, generate_samples, load_cached_inputs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASETS = ("PEMS", "AIR", "CA_NODE_SHIFT", "CA_MULTIYEAR_NODE_SHIFT")
DATA_FIELDS = ("begin_year", "end_year", "raw_data_path", "graph_path", "save_data_path",
               "x_len", "y_len", "steps_per_day", "pems_days", "train_rate", "val_rate", "test_rate")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default="STBP", help="STBP/STRAP/EAC/PECPM/STKEC/TS, or all")
    result.add_argument("--dataset", default="PEMS", help="one of four legacy datasets, or all")
    result.add_argument("--variant", choices=("unified", "raw", "both"), default="unified",
                        help="unified: standardize X only; raw: source X; both: paired runs")
    result.add_argument("--normalization-scope", choices=("split", "train"), default="split")
    result.add_argument("--gpuid", type=int, default=0, help="-1 explicitly selects CPU")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--epochs", "--epoch", dest="epoch", type=int)
    result.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    result.add_argument("--lr", type=float)
    result.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    result.add_argument("--output-dir", type=Path, default=ROOT / "runs_baselines")
    result.add_argument("--conf", type=Path, help="optional method/config overrides, CLI overrides last")
    result.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    result.add_argument("--smoke", action="store_true", help="real nodes, 4/2/2 windows, <=2 epochs per task")
    result.add_argument("--check-data", action="store_true")
    result.add_argument("--plan", action="store_true", help="print exact resolved configurations without training")
    return result


def merged(left, right):
    result = deepcopy(left)
    for name, value in right.items():
        result[name] = merged(result[name], value) if isinstance(value, dict) and isinstance(result.get(name), dict) else deepcopy(value)
    return result


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configuration(cli, model, dataset, variant):
    supplied = json.loads((HERE / "conf" / f"{dataset}.json").read_text(encoding="utf-8-sig"))
    config = {name: supplied[name] for name in DATA_FIELDS if name in supplied}
    config.update(baseline=model, method=model, dataset=dataset, variant=variant,
                  seed=cli.seed, gpuid=cli.gpuid, num_workers=cli.num_workers,
                  normalize_x=variant == "unified", normalization_scope=cli.normalization_scope,
                  target_scale="source", train_rate=.6, val_rate=.2, test_rate=.2,
                  val_test_mix=False, pems_days=31, steps_per_day=supplied.get("steps_per_day", 288),
                  x_len=12, y_len=12, epoch=200, batch_size=supplied["batch_size"],
                  lr=supplied["lr"], dropout=0.0, patience=30, weight_decay=.01, grad_clip=0.,
                  lr_schedule=None, batch_size_schedule=None,
                  cache_data=False, data_process=True, init=True, smoke=cli.smoke,
                  model={"in_channel": 12, "out_channel": 12, "hidden_channel": 192})
    module = implementation(model)
    config = merged(config, getattr(module, "DEFAULTS", {}))
    if hasattr(module, "defaults"):
        config = merged(config, module.defaults(dataset))
    if model not in {"STBP", "EAC"}:
        # These upstream networks read gcn.* or their method-specific width.
        config.pop("model", None)
    config.setdefault("fidelity_status", {
        "STBP": "local_source_predictor_and_native_pattern_tuning",
        "EAC": "pinned_official_predictor_and_native_low_rank_prompt_tuning",
        "STRAP": "pinned_official_three_library_model_common_protocol",
        "STKEC": "pinned_official_predictor_repaired_preprocessing_and_cluster_supervision",
    }.get(model, "see_baseline_provenance"))
    if cli.conf:
        config = merged(config, json.loads(cli.conf.read_text(encoding="utf-8-sig")))
    for name in ("epoch", "batch_size", "lr"):
        value = getattr(cli, name)
        if value is not None:
            config[name] = value
            if name in {"lr", "batch_size"}:
                config[name + "_schedule"] = None
    for setting in cli.set:
        if "=" not in setting:
            raise ValueError(f"--set needs KEY=JSON: {setting}")
        key, value = setting.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        target = config
        keys = key.split(".")
        for part in keys[:-1]:
            if not isinstance(target.get(part), dict):
                raise ValueError(f"Unknown nested setting: {key}")
            target = target[part]
        if keys[-1] not in target:
            raise ValueError(f"Unknown setting: {key}; use --conf to add an explicit method option")
        target[keys[-1]] = value
        if key in {"lr", "batch_size"}:
            config[key + "_schedule"] = None
    # A run label must always match the data protocol actually executed.
    config.update(baseline=model, method=model, dataset=dataset, variant=variant,
                  normalize_x=variant == "unified", target_scale="source", smoke=cli.smoke)
    if config["normalization_scope"] not in {"split", "train"}:
        raise ValueError("normalization_scope must be split or train")
    if cli.smoke:
        config.update(epoch=min(config["epoch"], 2), batch_size=2, num_workers=0, cache_data=False)
    if min(config["batch_size"], config["epoch"], config["lr"]) <= 0 or config["num_workers"] < 0:
        raise ValueError("batch-size, epochs and lr must be positive; workers must be nonnegative")
    if config["begin_year"] > config["end_year"]:
        raise ValueError("begin_year must not exceed end_year")
    if config["x_len"] != 12 or config["y_len"] != 12:
        raise ValueError("The integrated upstream baselines require the common 12 -> 12 protocol")
    if config["val_test_mix"]:
        raise ValueError("Baseline comparisons require chronological validation/test splits")
    config["output_root"] = str(cli.output_dir.resolve())
    config["protocol"] = {"split": "chronological_per_task", "rates": [config[k] for k in ("train_rate", "val_rate", "test_rate")],
                          "targets": "source", "input_preprocessing": "standardized_x" if variant == "unified" else "source_x",
                          "normalization_scope": config["normalization_scope"] if variant == "unified" else None,
                          "validation": "masked_MAE_batch_mean", "test": "global_masked_prefix_metrics",
                          "old_task_backtesting": False, "method_training_loss": "upstream_specific"}
    return config


def setup_device(args):
    if args.gpuid >= 0:
        if not torch.cuda.is_available() or args.gpuid >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA device unavailable; install CUDA PyTorch or explicitly pass --gpuid -1")
        return torch.device(f"cuda:{args.gpuid}")
    return torch.device("cpu")


def run(config):
    args = SimpleNamespace(**config)
    for name in ("raw_data_path", "graph_path", "save_data_path"):
        setattr(args, name, str(project_path(getattr(args, name))))
    missing = check_source_files(args)
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(map(str, missing)))
    args.device = setup_device(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.path = str(Path(args.output_root) / args.baseline / args.dataset / args.variant /
                    f"{stamp}_seed{args.seed}_{uuid.uuid4().hex[:6]}")
    Path(args.path).mkdir(parents=True)
    args.logger = logging.getLogger(args.path)
    args.logger.setLevel(logging.INFO)
    args.logger.propagate = False
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(args.path) / "train.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        args.logger.addHandler(handler)
    (Path(args.path) / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    args.logger.info("baseline=%s dataset=%s variant=%s device=%s run=%s settings=%s",
                     args.baseline, args.dataset, args.variant, args.device, args.path, config)
    adapter = implementation(args.baseline).Adapter(args)
    previous_raw, previous_inputs, previous_adjacency = None, None, None
    results = {}
    try:
        for year in range(args.begin_year, args.end_year + 1):
            args.year = year
            for name in ("lr", "batch_size"):
                schedule = config.get(name + "_schedule")
                if schedule and not (args.smoke and name == "batch_size"):
                    setattr(args, name, schedule[min(year - args.begin_year, len(schedule) - 1)])
            with np.load(Path(args.graph_path) / f"{year}_adj.npz", allow_pickle=False) as archive:
                adjacency = np.asarray(archive["x"], dtype=np.float32)
            with np.load(Path(args.raw_data_path) / f"{year}.npz", allow_pickle=False) as archive:
                raw = np.asarray(archive["x"], dtype=np.float32)
            if args.dataset == "PEMS" and args.pems_days > 0:
                raw = raw[:args.pems_days * args.steps_per_day]
            if raw.ndim != 2 or adjacency.shape != (raw.shape[1], raw.shape[1]):
                raise ValueError(f"Source and graph shape mismatch in task {year}")
            if not np.isfinite(raw).all() or not np.isfinite(adjacency).all():
                raise ValueError(f"Non-finite source/graph in task {year}")
            args.graph_size = adjacency.shape[0]
            args.expected_source_steps = len(raw)
            if year == args.begin_year:
                args.base_node_size = args.graph_size
            normalized = adjacency / (adjacency.sum(axis=1, keepdims=True) + 1e-6)
            args.adj = torch.as_tensor(normalized, dtype=torch.float32, device=args.device)
            cache = project_path(args.save_data_path) / f"x_{args.normalization_scope if args.normalize_x else 'raw'}_L12_H12" / f"{year}.npz"
            if args.data_process or args.smoke:
                destination = cache if args.cache_data and not args.smoke else None
                inputs = generate_samples(args.pems_days, destination, raw, nx.from_numpy_array(adjacency),
                                          train_rate=args.train_rate, val_rate=args.val_rate, test_rate=args.test_rate,
                                          normalize_x=args.normalize_x, normalization_scope=args.normalization_scope,
                                          x_len=args.x_len, y_len=args.y_len,
                                          sample_limits={"train": 4, "val": 2, "test": 2} if args.smoke else None)
            else:
                inputs = load_cached_inputs(cache, args)
            metadata = cache_metadata(inputs)
            args.logger.info("task=%s nodes=%s preprocessing=%s", year, args.graph_size, metadata)
            ctx = SimpleNamespace(args=args, logger=args.logger, inputs=inputs, adj=args.adj,
                                  raw_adj=adjacency, raw_data=raw,
                                  train_raw_data=raw[:int(len(raw) * args.train_rate)],
                                  previous_raw_data=previous_raw, previous_inputs=previous_inputs,
                                  previous_raw_adj=previous_adjacency)
            results[str(year)] = fit_task(adapter, ctx)
            (Path(args.path) / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            previous_raw, previous_inputs, previous_adjacency = raw, inputs, adjacency
        average = {horizon: {metric: float(np.mean([row["metrics"][horizon][metric] for row in results.values()]))
                             for metric in ("MAE", "RMSE", "MAPE")}
                   for horizon in next(iter(results.values()))["metrics"]}
        report = {"baseline": args.baseline, "dataset": args.dataset, "variant": args.variant,
                  "run": args.path, "smoke": args.smoke, "tasks": list(results), "average": average,
                  "protocol": config["protocol"], "fidelity_status": config.get("fidelity_status", "see_baseline_provenance")}
        (Path(args.path) / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        args.logger.info("COMPLETED %s", report)
        return report
    finally:
        for handler in list(args.logger.handlers):
            handler.close()
            args.logger.removeHandler(handler)
        del adapter, previous_raw, previous_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    cli = parser().parse_args()
    models = MODELS if cli.model.lower() == "all" else (canonical_model(cli.model),)
    dataset = cli.dataset.upper().replace("-", "_")
    datasets = DATASETS if dataset == "ALL" else (dataset,)
    if any(item not in DATASETS for item in datasets):
        raise ValueError(f"Unknown dataset; choose {DATASETS}")
    variants = ("unified", "raw") if cli.variant == "both" else (cli.variant,)
    configs = [configuration(cli, model, dataset, variant)
               for model in models for dataset in datasets for variant in variants]
    if cli.plan:
        print(json.dumps({"runs": len(configs), "configurations": configs}, ensure_ascii=False, indent=2))
        return
    for config in configs:
        args = SimpleNamespace(**config)
        args.raw_data_path, args.graph_path = str(project_path(args.raw_data_path)), str(project_path(args.graph_path))
        missing = check_source_files(args)
        if missing:
            raise FileNotFoundError("Missing source files:\n" + "\n".join(map(str, missing)))
    if cli.check_data:
        print(f"PASS: all source files exist for {len(configs)} configurations")
        return
    reports = []
    for index, config in enumerate(configs):
        print(f"RUN {index + 1}/{len(configs)} {config['baseline']} {config['dataset']} {config['variant']}", flush=True)
        reports.append(run(config))
    print(json.dumps({"status": "passed", "runs": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
