"""Shared utilities for the Madrid continual forecasting protocol."""

import argparse

import json

import os

from pathlib import Path

ROOT = Path(__file__).resolve().parent

METHOD = "STBP_MADRID_2022_2026_MMD"

DEFAULTS = {
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
    "cache_data": True, "data_process": 1, "init": True, "train": 1,
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

def configuration(cli, dataset):
    source = cli.conf or ROOT / "conf" / f"{dataset}.json"
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
    config["method"] = METHOD
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
    return path if path.is_absolute() else ROOT / path
