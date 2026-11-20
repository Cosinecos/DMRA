"""Train Dual Memories and Residual Adaptation on a sequence of sensor networks."""

import argparse
from datetime import datetime
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

from dmra.runtime.data import check_source_files, generate_samples, load_cached_inputs, source_shape
from dmra.model import model_class, method_for, validate_embedding_config
from dmra.runtime.trainer import run_task


ROOT = Path(__file__).resolve().parents[1]
METHOD = "DMRA"
DATASETS = ("PEMS", "AIR", "CA_NODE_SHIFT", "CA_MULTIYEAR_NODE_SHIFT")
DEFAULTS = {
    "embedding_variant": "dmra",
    "bank_temperature": 0.3,
    "bank_qk_norm_eps": 1e-8,
    "week_interaction_rank": 16,
    "gcn_adapter_rank": 16,
    "gcn_adapter_lora_rank": 4,
    "use_gcn_adapter_lora": True,
    "seed": 42, "gpuid": 0, "num_workers": 0 if os.name == "nt" else 4,
    "normalize_x": False, "normalization_scope": "split", "use_residual_norm": True,
    "use_mae_loss": True, "use_freq_loss": True, "use_bank_ortho_loss": True,
    "use_node_stability_loss": True, "use_distillation_loss": True,
    "lambda_mae": 1.0, "lambda_freq": 0.0, "lambda_bank_ortho": 0.0,
    "lambda_u": 0.0, "lambda_dist": 0.0, "freq_remove_dc": True,
    "ablate_v6_scale_head": False, "ablate_v6_season": False,
    "ablate_v6_time_features": False, "ablate_v6_node_graph": False,
    "learnable_node_graph_scale": True, "steps_per_day": 288, "pems_days": 31,
    "train_rate": .6, "val_rate": .2, "test_rate": .2, "val_test_mix": False,
    "cache_data": False, "data_process": 1, "init": True, "train": 1,
    "use_lora": True, "retrain_match_capacity": False,
    "load_first_year": 0, "first_year_model_path": "", "checkpoint_path": "",
    "profile_v6_parts": False, "profile_v6_epoch": -1,
    "pred_dump": False, "pred_dump_interval": 10, "pred_dump_samples": 2,
    "pred_dump_nodes": 5, "pred_dump_plots": True,
    "smoke": False, "patience": 30, "paral": 0, "res_norm_eps": 1e-5,
    "model": {"in_channel": 12, "out_channel": 12, "hidden_channel": 192},
}
BOOL_FIELDS = {name for name, value in DEFAULTS.items() if isinstance(value, bool)}


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "on", "yes", "1"}:
        return True
    if str(value).lower() in {"false", "off", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false or on/off")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", default="PEMS", help="PEMS, AIR, CA_NODE_SHIFT, CA_MULTIYEAR_NODE_SHIFT, or all")
    result.add_argument("--conf", type=Path, help="JSON config; CLI options override it")
    result.add_argument("--gcn-adapter-rank", type=int, help="positive bottleneck width; default16")
    result.add_argument("--gcn-adapter-lora-rank", type=int, help="annual Adapter matrix LoRA rank; default4")
    result.add_argument("--gcn-adapter-lora", dest="use_gcn_adapter_lora", type=boolean, help="enable Adapter annual LoRA; base Adapter stays frozen in incremental tasks")
    result.add_argument("--bank-temperature", type=float, help="fixed positive cosine retrieval temperature; default0.3")
    result.add_argument("--bank-qk-norm-eps", type=float, help="positive Q/K normalization epsilon; default1e-8")
    result.add_argument("--week-interaction-rank", type=int, help="new TOD/DOW interaction embedding rank; default16")
    result.add_argument("--gpuid", type=int)
    result.add_argument("--seed", type=int)
    result.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int)
    result.add_argument("--epochs", "--epoch", dest="epoch", type=int)
    result.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    result.add_argument("--normalize-x", dest="normalize_x", type=boolean)
    result.add_argument("--normalization-scope", choices=("split", "train"))
    result.add_argument("--residual-norm", dest="use_residual_norm", type=boolean)
    result.add_argument("--fixed-stats", dest="ablate_v6_scale_head", type=boolean)
    for option, field in (("mae-loss", "use_mae_loss"), ("freq-loss", "use_freq_loss"),
                          ("bank-ortho-loss", "use_bank_ortho_loss"),
                          ("node-stability-loss", "use_node_stability_loss"),
                          ("distillation-loss", "use_distillation_loss")):
        result.add_argument("--" + option, dest=field, type=boolean)
    result.add_argument("--data-process", "--data_process", dest="data_process", type=boolean)
    result.add_argument("--train", type=boolean)
    result.add_argument("--init", type=boolean)
    result.add_argument("--lora", dest="use_lora", type=boolean,
                        help="add annual LoRA adapters; use false with --init false for retraining without LoRA")
    result.add_argument("--retrain-match-capacity", type=boolean,
                        help="with --init false --lora true, freshly create all LoRA years to match incremental model size")
    result.add_argument("--load-first-year", "--load_first_year", dest="load_first_year", type=boolean)
    result.add_argument("--first-year-model-path", "--first_year_model_path", dest="first_year_model_path")
    result.add_argument("--output-dir", type=Path, help="override result root; data are never written here")
    result.add_argument("--smoke", action="store_true", help="real nodes/losses; tiny windows and 2 epochs/task")
    result.add_argument("--check-data", action="store_true", help="only check all required task files")
    result.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                        help="override any existing config field, e.g. lambda_freq=0.01 or model.hidden_channel=192")
    return result


def configuration(cli, dataset):
    source = cli.conf or ROOT / "dmra" / "config" / f"{dataset}.json"
    if not source.is_absolute():
        source = ROOT / source
    config = json.loads(json.dumps(DEFAULTS))
    supplied = json.loads(source.read_text(encoding="utf-8-sig"))
    model = dict(config["model"], **supplied.get("model", {}))
    config.update(supplied)
    config["model"] = model
    for key, value in vars(cli).items():
        if key in {"conf", "dataset", "output_dir", "check_data", "set", "smoke"}:
            continue
        if value is not None:
            config[key] = value
    for setting in cli.set:
        if "=" not in setting:
            raise ValueError(f"--set needs KEY=VALUE: {setting}")
        key, value = setting.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        target = config
        path = key.split(".")
        for part in path[:-1]:
            if not isinstance(target.get(part), dict):
                raise ValueError(f"Unknown nested setting: {key}")
            target = target[part]
        if path[-1] not in target:
            raise ValueError(f"Unknown setting: {key}; add it explicitly to the JSON config first")
        target[path[-1]] = value
    for name in BOOL_FIELDS:
        config[name] = boolean(config[name])
    for name in ("data_process", "init", "train", "load_first_year"):
        config[name] = boolean(config[name])
    if config["retrain_match_capacity"] and (config["init"] or not config["use_lora"]):
        raise ValueError("retrain_match_capacity requires --init false and --lora true")
    validate_embedding_config(config)
    config["method"] = method_for(config["embedding_variant"])
    config["dataset"] = supplied.get("dataset", dataset)
    config["smoke"] = bool(cli.smoke)
    if cli.smoke:
        config.update(epoch=min(int(config["epoch"]), 2), batch_size=2, num_workers=0, cache_data=False)
    if cli.output_dir is not None:
        config["model_path"] = str(cli.output_dir / config["dataset"])
    if config["normalization_scope"] not in {"split", "train"}:
        raise ValueError("normalization_scope must be split or train")
    if config["batch_size"] <= 0 or config["epoch"] <= 0 or config["num_workers"] < 0:
        raise ValueError("batch_size/epoch must be positive and num_workers must be nonnegative")
    if abs(sum(float(config[k]) for k in ("train_rate", "val_rate", "test_rate")) - 1) > 1e-6:
        raise ValueError("train_rate + val_rate + test_rate must equal 1")
    for name in ("lambda_mae", "lambda_freq", "lambda_bank_ortho", "lambda_u", "lambda_dist"):
        if float(config[name]) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if config["end_year"] < config["begin_year"]:
        raise ValueError("end_year must not precede begin_year")
    effective_padding = (int(config["standard_gcn_tcn_kernel_size"]) - 1) * int(config["standard_gcn_tcn_dilation"])
    if effective_padding % 2:
        raise ValueError("Convolution kernel/dilation must preserve the hidden width: (kernel-1)*dilation must be even")
    return config


def project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data" and os.environ.get("DMRA_DATA_ROOT"):
        return Path(os.environ["DMRA_DATA_ROOT"]).expanduser().resolve().joinpath(*path.parts[1:])
    return ROOT / path


def run(config):
    args = SimpleNamespace(**config)
    for field in ("raw_data_path", "graph_path", "save_data_path", "model_path"):
        setattr(args, field, str(project_path(getattr(args, field))))
    missing = check_source_files(args)
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(map(str, missing)))
    if args.gpuid >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install CUDA PyTorch or explicitly use --gpuid -1 for CPU")
        if args.gpuid >= torch.cuda.device_count():
            raise ValueError(f"GPU {args.gpuid} is unavailable; visible count={torch.cuda.device_count()}")
        args.device = torch.device(f"cuda:{args.gpuid}")
    else:
        args.device = torch.device("cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.path = str(Path(args.model_path) / f"{stamp}_seed{args.seed}_{uuid.uuid4().hex[:6]}")
    Path(args.path).mkdir(parents=True)
    args.logger = logging.getLogger(args.path)
    args.logger.setLevel(logging.INFO)
    args.logger.propagate = False
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(args.path) / "train.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        args.logger.addHandler(handler)
    args.methods = {args.method: model_class(args.embedding_variant)}
    (Path(args.path) / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    args.logger.info("device=%s run=%s settings=%s", args.device, args.path, config)
    if not args.use_residual_norm:
        args.logger.info("Residual normalization and restoration are BOTH bypassed; scale_head is not called")
    args.logger.info("embedding=%s bank_retrieval=cosine_shared_key_value week_interaction=new_embeddings gcn_terminal_adapter=true adapter_annual_matrix_lora=true calendar=task_relative_phase; explicit natural weekday alignment is not changed", args.embedding_variant)
    results = {}
    args.graph_size_list = []
    for year in range(args.begin_year, args.end_year + 1):
        args.year = year
        args.prev_adj = getattr(args, "adj", None)
        with np.load(Path(args.graph_path) / f"{year}_adj.npz") as archive:
            adjacency = archive["x"]
        args.graph_size = adjacency.shape[0]
        source_steps, source_nodes = source_shape(Path(args.raw_data_path) / f"{year}.npz")
        if source_nodes != args.graph_size or adjacency.shape != (source_nodes, source_nodes):
            raise ValueError(f"Source and square graph node counts differ in task {year}")
        args.expected_source_steps = (min(source_steps, args.pems_days * args.steps_per_day)
                                      if args.dataset == "PEMS" and args.pems_days > 0 else source_steps)
        if year == args.begin_year:
            args.base_node_size = args.graph_size
        args.graph_size_list.append(args.graph_size)
        normalized = adjacency / (adjacency.sum(axis=1, keepdims=True) + 1e-6)
        args.adj = torch.as_tensor(normalized, dtype=torch.float32, device=args.device)
        scope = args.normalization_scope if args.normalize_x else "raw"
        cache = Path(args.save_data_path) / f"x_{scope}_L{args.x_len}_H{args.y_len}" / f"{year}.npz"
        if args.data_process or args.smoke:
            with np.load(Path(args.raw_data_path) / f"{year}.npz") as archive:
                raw = archive["x"]
            if args.dataset == "PEMS" and args.pems_days > 0:
                raw = raw[:args.pems_days * args.steps_per_day]
            output = cache if args.cache_data and not args.smoke else None
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
            inputs = generate_samples(args.pems_days, output, raw, nx.from_numpy_array(adjacency),
                                      train_rate=args.train_rate, val_rate=args.val_rate, test_rate=args.test_rate,
                                      val_test_mix=args.val_test_mix, normalize_x=args.normalize_x,
                                      normalization_scope=args.normalization_scope,
                                      x_len=args.x_len, y_len=args.y_len,
                                      sample_limits={"train": 4, "val": 2, "test": 2} if args.smoke else None)
            del raw
        else:
            inputs = load_cached_inputs(cache, args)
        args.logger.info("task=%s nodes=%s samples=%s", year, args.graph_size,
                         {split: len(inputs[split + "_x"]) for split in ("train", "val", "test")})
        results[str(year)] = run_task(inputs, args)
        (Path(args.path) / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        del inputs
    average = {}
    for horizon in results[str(args.begin_year)]["metrics"]:
        average[horizon] = {metric: float(np.mean([row["metrics"][horizon][metric] for row in results.values()]))
                            for metric in ("MAE", "RMSE", "MAPE")}
    report = {"dataset": args.dataset, "run": str(Path(args.path).relative_to(ROOT)) if Path(args.path).is_relative_to(ROOT) else str(args.path),
              "smoke": args.smoke, "tasks": list(results), "average": average}
    (Path(args.path) / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.logger.info("COMPLETED %s", report)
    for handler in list(args.logger.handlers):
        handler.close()
        args.logger.removeHandler(handler)
    return report


def main():
    cli = parser().parse_args()
    dataset = cli.dataset.upper().replace("-", "_")
    selected = list(DATASETS) if dataset == "ALL" else [dataset]
    if any(name not in DATASETS for name in selected):
        raise ValueError(f"Unknown dataset: {cli.dataset}; choose {DATASETS} or all")
    if cli.conf is not None and len(selected) > 1:
        raise ValueError("Use one --conf or --dataset all, not both")
    configs = [configuration(cli, name) for name in selected]
    for config in configs:
        args = SimpleNamespace(**config)
        args.raw_data_path = str(project_path(args.raw_data_path))
        args.graph_path = str(project_path(args.graph_path))
        missing = check_source_files(args)
        if missing:
            raise FileNotFoundError("Missing source files:\n" + "\n".join(map(str, missing)))
    if cli.check_data:
        print(f"PASS: {len(configs)} dataset configurations have all required source files")
        return
    reports = [run(config) for config in configs]
    print(json.dumps({"status": "passed", "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
