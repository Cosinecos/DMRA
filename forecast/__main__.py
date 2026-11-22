"""Train the conventional baselines under the full-data DMRA protocol."""

import argparse
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

import numpy as np
import torch

from . import MODELS, canonical_model, implementation
from .data import TaskData
from .trainer import fit_task


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("PEMS", "AIR", "CA_NODE_SHIFT", "CA_MULTIYEAR_NODE_SHIFT", "MADRID")
DATASET_ALIASES = {
    "CA_S": "CA_NODE_SHIFT",
    "CA_M": "CA_MULTIYEAR_NODE_SHIFT",
    "MADRID_2022_2026_MMD": "MADRID",
}
DATASET_FIELDS = (
    "begin_year", "end_year", "raw_data_path", "graph_path", "steps_per_day", "pems_days",
)
TRAINING = {
    "TQNET": {"lr": .0001},
    "GWNET": {"lr": .001, "weight_decay": .0001, "grad_clip": 5.,
              "batch_size": 16, "microbatch_size": 16},
    "STID": {"lr": .001, "weight_decay": .0001},
    "PATCHTST": {"lr": .0001},
}


def canonical_dataset(name):
    name = name.strip().upper().replace("-", "_")
    name = DATASET_ALIASES.get(name, name)
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose PEMS, AIR, CA-S, CA-M, or Madrid")
    return name


def selection(value, all_names, canonical):
    if value.strip().lower() == "all":
        return all_names
    return tuple(dict.fromkeys(canonical(name) for name in value.split(",")))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", help="TQNet, GWNet, STID, PatchTST, a comma-separated list, or all")
    parser.add_argument("--dataset", default="all", help="PEMS, AIR, CA-S, CA-M, Madrid, a comma-separated list, or all")
    parser.add_argument("--gpuid", type=int, default=0, help="CUDA device index, or -1 for CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--begin-year", type=int, help="First stage ID for a single dataset")
    parser.add_argument("--end-year", type=int, help="Last stage ID for a single dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs_forecast_fullft")
    parser.add_argument("--smoke", action="store_true", help="Use all real nodes and only 4/2/2 train/validation/test windows per stage")
    parser.add_argument("--check-data", action="store_true", help="Prepare and check the input files, then exit")
    cli = parser.parse_args()
    if cli.epochs < 1 or cli.patience < 0 or cli.num_workers < 0:
        parser.error("epochs must be positive and patience/num-workers must be nonnegative")
    if any(value is not None and value < 1 for value in (cli.batch_size, cli.microbatch_size)):
        parser.error("batch-size and microbatch-size must be positive")
    if cli.gpuid < -1:
        parser.error("gpuid must be -1 for CPU or a nonnegative CUDA device index")
    try:
        models = selection(cli.model, MODELS, canonical_model)
        datasets = selection(cli.dataset, DATASETS, canonical_dataset)
        if len(datasets) > 1 and (cli.begin_year is not None or cli.end_year is not None):
            raise ValueError("Stage ranges require exactly one dataset")
        configs = [configuration(cli, model, dataset) for model in models for dataset in datasets]
    except ValueError as error:
        parser.error(str(error))
    return cli, configs


def configuration(cli, model, dataset):
    tag = "MADRID_2022_2026_MMD" if dataset == "MADRID" else dataset
    path = ROOT / "dmra" / "config" / f"{tag}.json"
    dataset_config = json.loads(path.read_text(encoding="utf-8-sig"))
    config = {name: dataset_config[name] for name in DATASET_FIELDS}
    first, last = config["begin_year"], config["end_year"]
    if cli.begin_year is not None:
        config["begin_year"] = cli.begin_year
    if cli.end_year is not None:
        config["end_year"] = cli.end_year
    if not first <= config["begin_year"] <= config["end_year"] <= last:
        raise ValueError(f"{dataset} stages must be in [{first}, {last}]")

    config.update(model_name=model, dataset=dataset, variant="raw", strategy="full_ft",
                  model=implementation(model).DEFAULTS.copy(), seed=cli.seed,
                  smoke=cli.smoke, epoch=cli.epochs, patience=cli.patience,
                  num_workers=cli.num_workers, batch_size=64 if dataset == "PEMS" else 128,
                  microbatch_size=2, weight_decay=0., grad_clip=0., log_every=20,
                  x_len=12, y_len=12, train_rate=.6, val_rate=.2, test_rate=.2)
    config.update(TRAINING[model])
    if cli.batch_size is not None:
        config["batch_size"] = cli.batch_size
    if cli.microbatch_size is not None:
        config["microbatch_size"] = cli.microbatch_size
    if cli.smoke:
        config.update(epoch=min(config["epoch"], 2), batch_size=2, num_workers=0,
                      microbatch_size=min(config["microbatch_size"], 2), log_every=1)
    return config


def check_sources(config):
    raw_dir = ROOT / config["raw_data_path"]
    graph_dir = ROOT / config["graph_path"]
    for year in range(config["begin_year"], config["end_year"] + 1):
        for path in (raw_dir / f"{year}.npz", graph_dir / f"{year}_adj.npz"):
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared source: {path}")


def device_for(gpuid):
    if gpuid == -1:
        return torch.device("cpu")
    if not torch.cuda.is_available() or gpuid >= torch.cuda.device_count():
        raise RuntimeError("Requested CUDA device is unavailable; use --gpuid -1 for CPU")
    return torch.device(f"cuda:{gpuid}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(config, output_dir, device):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (output_dir / config["model_name"] / config["dataset"] / "raw" /
               f'{stamp}_seed{config["seed"]}_{uuid4().hex[:6]}')
    run_dir.mkdir(parents=True)
    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = (logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)

    results = {}
    try:
        write_json(run_dir / "config.json", config)
        logger.info("baseline=%s dataset=%s variant=raw device=%s strategy=full_ft settings=%s",
                    config["model_name"], config["dataset"], device, config)
        with TemporaryDirectory(prefix="dmra-forecast-") as checkpoint_dir:
            previous = None
            for year in range(config["begin_year"], config["end_year"] + 1):
                started = time.perf_counter()
                task = TaskData(ROOT, config, year)
                preprocessing_seconds = time.perf_counter() - started
                logger.info("task=%s preprocessing_finished seconds=%.3f nodes=%s samples=%s",
                            task.label, preprocessing_seconds, task.nodes, task.metadata["samples"])
                checkpoint = Path(checkpoint_dir) / f"{year}.pkl"
                result = fit_task(task, config, checkpoint, logger, device, previous)
                result["preprocessing_seconds"] = preprocessing_seconds
                results[str(year)] = result
                write_json(run_dir / "results.json", results)
                if previous is not None:
                    previous.unlink()
                previous = checkpoint
                del task
                gc.collect()

        average = {}
        metrics = next(iter(results.values()))["metrics"]
        names = ("MAE", "RMSE", "MAPE", "WAPE") if config["dataset"] == "MADRID" else ("MAE", "RMSE", "MAPE")
        for prefix in metrics:
            averages = {}
            for name in names:
                values = [row["metrics"][prefix][name] for row in results.values()]
                observed = [value for value in values if value is not None]
                averages[name] = float(np.mean(observed)) if observed else None
            average[prefix] = averages
        report = dict(model=config["model_name"], dataset=config["dataset"], variant="raw",
                      strategy="full_ft", smoke=config["smoke"], tasks=list(results), average=average,
                      run=str(run_dir))
        write_json(run_dir / "summary.json", report)
        logger.info("COMPLETED %s", report)
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main():
    cli, configs = parse_args()
    subprocess.run([sys.executable, str(ROOT / "prepare_data.py")], check=True)
    by_dataset = {config["dataset"]: config for config in configs}
    for config in by_dataset.values():
        check_sources(config)
    if cli.check_data:
        print(f"Data ready for {len(configs)} conventional model/dataset combinations.")
        return
    device = device_for(cli.gpuid)
    for config in configs:
        run(config, cli.output_dir, device)


if __name__ == "__main__":
    main()
