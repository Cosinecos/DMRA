"""Stage-wise optimization, validation, and evaluation for DMRA."""
import copy
import csv
from datetime import datetime
import json
import os
import os.path as osp
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from .data import SpatioTemporalDataset
from .checkpoint import load_best_model, save_checkpoint
from .metrics import MAE_torch, masked_mae_np, masked_mse_np, masked_mape_np
from dmra.model import model_class

def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def _check_finite(tensor, name, args, epoch=None, batch_idx=None):
    if torch.isfinite(tensor).all():
        return
    location = f"year={getattr(args, 'year', '?')}"
    if epoch is not None:
        location += f", epoch={epoch}"
    if batch_idx is not None:
        location += f", batch={batch_idx}"
    raise FloatingPointError(f"Non-finite values in {name} ({location}).")

def spectral_amplitude_loss(pred, target, null_val=0.0, remove_dc=True):
    valid = ~torch.isclose(target, torch.tensor(null_val, device=target.device, dtype=target.dtype))
    valid_series = valid.all(dim=-1)
    if not valid_series.any():
        return pred.sum() * 0.0

    pred_valid = pred[valid_series]
    target_valid = target[valid_series]
    pred_freq = torch.fft.rfft(pred_valid, dim=-1, norm="ortho")
    target_freq = torch.fft.rfft(target_valid, dim=-1, norm="ortho")
    pred_amp = torch.abs(pred_freq)
    target_amp = torch.abs(target_freq)
    if remove_dc and pred_amp.shape[-1] > 1:
        pred_amp = pred_amp[..., 1:]
        target_amp = target_amp[..., 1:]
    return torch.mean(torch.abs(pred_amp - target_amp))

def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False

def _prediction_dump_enabled(args):
    return _as_bool(getattr(args, "pred_dump", False))

def _copy_if_needed(src, dst):
    if src == dst:
        return
    shutil.copyfile(src, dst)

def _write_prediction_csv(path, split, epoch, pred_dense, truth_dense, x_dense, max_samples, max_nodes):
    pred_np = pred_dense.detach().cpu().numpy()
    truth_np = truth_dense.detach().cpu().numpy()
    x_np = x_dense.detach().cpu().numpy()
    sample_count = min(max_samples, pred_np.shape[0])
    node_count = min(max_nodes, pred_np.shape[1])

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "split",
            "epoch",
            "sample_idx",
            "node_idx",
            "horizon",
            "input_last",
            "label",
            "prediction",
            "abs_error",
        ])
        for sample_idx in range(sample_count):
            for node_idx in range(node_count):
                input_last = float(x_np[sample_idx, node_idx, -1])
                for horizon in range(pred_np.shape[-1]):
                    label = float(truth_np[sample_idx, node_idx, horizon])
                    prediction = float(pred_np[sample_idx, node_idx, horizon])
                    writer.writerow([
                        split,
                        epoch,
                        sample_idx,
                        node_idx,
                        horizon + 1,
                        f"{input_last:.6f}",
                        f"{label:.6f}",
                        f"{prediction:.6f}",
                        f"{abs(prediction - label):.6f}",
                    ])

def _write_prediction_plot(path, split, epoch, pred_dense, truth_dense, x_dense):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    pred_np = pred_dense.detach().cpu().numpy()
    truth_np = truth_dense.detach().cpu().numpy()
    x_np = x_dense.detach().cpu().numpy()
    horizon = np.arange(1, pred_np.shape[-1] + 1)
    input_last = float(x_np[0, 0, -1])

    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(horizon, truth_np[0, 0], marker="o", linewidth=2.0, label="Label", color="#111827")
    plt.plot(horizon, pred_np[0, 0], marker="s", linewidth=1.8, label="Prediction", color="#2563eb")
    plt.axhline(input_last, linestyle="--", linewidth=1.2, label="Input last", color="#6b7280")
    plt.title(f"{split} prediction debug | year {epoch[0]} epoch {epoch[1]}")
    plt.xlabel("Horizon")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True

def _dump_prediction_split(model, args, loader, split, adj, epoch, output_dir):
    data = next(iter(loader))
    data = data.to(args.device, non_blocking=True)
    pred = model(data, adj)
    pred_dense, _ = to_dense_batch(pred, batch=data.batch)
    truth_dense, _ = to_dense_batch(data.y, batch=data.batch)
    x_dense, _ = to_dense_batch(data.x, batch=data.batch)

    max_samples = max(1, int(getattr(args, "pred_dump_samples", 2)))
    max_nodes = max(1, int(getattr(args, "pred_dump_nodes", 5)))
    prefix = f"epoch_{epoch:04d}_{split}"
    csv_path = osp.join(output_dir, f"{prefix}.csv")
    _write_prediction_csv(csv_path, split, epoch, pred_dense, truth_dense, x_dense, max_samples, max_nodes)
    _copy_if_needed(csv_path, osp.join(output_dir, f"latest_{split}.csv"))

    if _as_bool(getattr(args, "pred_dump_plots", True)):
        plot_path = osp.join(output_dir, f"{prefix}.png")
        if _write_prediction_plot(plot_path, split, (args.year, epoch), pred_dense, truth_dense, x_dense):
            _copy_if_needed(plot_path, osp.join(output_dir, f"latest_{split}.png"))

    mae = masked_mae_np(truth_dense.cpu().numpy(), pred_dense.cpu().numpy(), 0)
    return float(mae)

def _maybe_dump_predictions(model, args, epoch, train_loader, test_loader):
    if not _prediction_dump_enabled(args):
        return
    interval = max(1, int(getattr(args, "pred_dump_interval", 10)))
    if epoch != 0 and (epoch + 1) % interval != 0:
        return

    was_training = model.training
    model.eval()
    output_dir = osp.join(args.path, "prediction_debug", f"year_{args.year}")
    mkdirs(output_dir)
    with torch.no_grad():
        train_mae = _dump_prediction_split(model, args, train_loader, "train", args.sub_adj, epoch, output_dir)
        test_mae = _dump_prediction_split(model, args, test_loader, "test", args.adj, epoch, output_dir)

    with open(osp.join(output_dir, "latest_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "epoch", "train_first_batch_mae", "test_first_batch_mae"])
        writer.writerow([args.year, epoch, f"{train_mae:.6f}", f"{test_mae:.6f}"])

    args.logger.info(
        "[prediction_debug] year %s epoch %s train_first_batch_mae %.4f test_first_batch_mae %.4f -> %s",
        args.year,
        epoch,
        train_mae,
        test_mae,
        output_dir,
    )
    if was_training:
        model.train()


def _enabled(args, flag, weight, default_weight=0.0):
    return bool(getattr(args, flag, True)) and float(getattr(args, weight, default_weight)) > 0


def needs_teacher(args):
    return (_enabled(args, "use_node_stability_loss", "lambda_u")
            or _enabled(args, "use_distillation_loss", "lambda_dist"))


def compute_loss(pred, target, model, args, batch=None, teacher_model=None):
    """Only execute enabled, positively weighted loss terms.

    Components contain unweighted tensors. Teacher terms are available only in
    incremental tasks; a first task with no active objective is rejected.
    """
    components = {}
    weights = {}
    if _enabled(args, "use_mae_loss", "lambda_mae", 1.0):
        components["mae"] = MAE_torch(pred, target, 0.0)
        weights["mae"] = float(getattr(args, "lambda_mae", 1.0))
    if _enabled(args, "use_freq_loss", "lambda_freq"):
        components["freq"] = spectral_amplitude_loss(
            pred.reshape(-1, args.graph_size, args.y_len),
            target.reshape(-1, args.graph_size, args.y_len),
            0.0, getattr(args, "freq_remove_dc", True),
        )
        weights["freq"] = float(args.lambda_freq)
    if _enabled(args, "use_bank_ortho_loss", "lambda_bank_ortho"):
        bank = torch.nn.functional.normalize(model.trend_branch.bank, dim=-1)
        count = bank.shape[0]
        gram = bank @ bank.T
        components["bank_ortho"] = ((gram - torch.eye(count, device=bank.device)) ** 2).sum() / count ** 2
        weights["bank_ortho"] = float(args.lambda_bank_ortho)
    if teacher_model is not None:
        old_nodes = teacher_model.num_nodes
        if _enabled(args, "use_node_stability_loss", "lambda_u"):
            components["node_stability"] = torch.mean(
                (model.node_embedding[:old_nodes] - teacher_model.node_embedding.detach()) ** 2
            )
            weights["node_stability"] = float(args.lambda_u)
        if _enabled(args, "use_distillation_loss", "lambda_dist"):
            if batch is None:
                raise ValueError("Distillation requires the current batch and previous adjacency")
            old_x = batch.x.reshape(-1, args.graph_size, args.x_len)[:, :old_nodes]
            x_tod = batch.x_tod.reshape(-1, args.x_len)
            x_dow = batch.x_dow.reshape(-1, args.x_len)
            y_tod = batch.y_tod.reshape(-1, args.y_len)
            y_dow = batch.y_dow.reshape(-1, args.y_len)
            adj = args.prev_adj[:old_nodes, :old_nodes]
            with torch.no_grad():
                reference = teacher_model.forward_dense(old_x, x_tod, x_dow, y_tod, y_dow, adj)
            student = model.forward_dense(old_x, x_tod, x_dow, y_tod, y_dow, adj)
            components["distillation"] = torch.mean((student - reference) ** 2)
            weights["distillation"] = float(args.lambda_dist)
    if not components:
        raise ValueError("No active training loss: enable MAE, frequency, or bank loss for the first task")
    total = sum(weights[name] * value for name, value in components.items())
    return total, components


def _loader(inputs, split, args, shuffle=False):
    dataset = SpatioTemporalDataset(inputs, split, steps_per_day=args.steps_per_day)
    if not len(dataset):
        raise ValueError(f"Empty {split} dataset for task {args.year}")
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=args.device.type == "cuda")


def evaluate(model, args, loader):
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            pred = model(batch, args.adj)
            _check_finite(pred, "test prediction", args)
            predictions.append(pred.reshape(-1, args.graph_size, args.y_len).cpu().numpy())
            truths.append(batch.y.reshape(-1, args.graph_size, args.y_len).cpu().numpy())
    truth, pred = np.concatenate(truths), np.concatenate(predictions)
    prefixes = []
    result = {}
    for horizon in range(1, args.y_len + 1):
        values = {
            "MAE": float(masked_mae_np(truth[..., :horizon], pred[..., :horizon], 0)),
            "RMSE": float(masked_mse_np(truth[..., :horizon], pred[..., :horizon], 0) ** .5),
            "MAPE": float(masked_mape_np(truth[..., :horizon], pred[..., :horizon], 0)),
        }
        prefixes.append(values)
        if horizon in {3, 6, args.y_len}:
            result[str(horizon)] = values
    result["Avg"] = {key: float(np.mean([row[key] for row in prefixes])) for key in prefixes[0]}
    for horizon, values in result.items():
        args.logger.info("test task=%s prefix=%s %s", args.year, horizon, values)
    return result


def run_task(inputs, args):
    task_dir = Path(args.path) / str(args.year)
    task_dir.mkdir(parents=True, exist_ok=True)
    args.sub_adj = args.adj
    test_loader = _loader(inputs, "test", args)
    if not args.train or (args.load_first_year and args.year == args.begin_year):
        model, _ = load_best_model(args)
        return {"metrics": evaluate(model, args, test_loader), "epochs": 0}
    train_loader = _loader(inputs, "train", args, shuffle=True)
    val_loader = _loader(inputs, "val", args)
    teacher = None
    if args.init and args.year > args.begin_year:
        model, _ = load_best_model(args)
        if needs_teacher(args):
            teacher = copy.deepcopy(model).to(args.device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad = False
        model.expand_adaptive_params(args.graph_size, args.adj)
        model.configure_incremental_training()
    else:
        model = model_class(args.embedding_variant)(args).to(args.device)
        model.expand_adaptive_params(args.graph_size, args.adj)
    if hasattr(model, "fit_state_calibration"):
        if not (args.init and args.year > args.begin_year):
            # Input windows are exactly those seen by the predictor. No label,
            # validation/test statistics, or later-year values enter this fit.
            model.fit_state_calibration(inputs["train_x"].transpose(0, 2, 1),
                                        chunk_size=args.state_calibration_chunk_size)
        args.logger.info("state_calibration=training_windows frozen_across_incremental_tasks=%s center=%s scale=%s",
                         bool(args.init), model.trend_branch.state_center.detach().cpu().tolist(),
                         model.trend_branch.state_scale.detach().cpu().tolist())
    model.count_parameters()
    args.logger.info("teacher=%s residual_norm=%s normalize_x=%s normalization_scope=%s lora_enabled=%s lora_years=%s retrain_match_capacity=%s",
                     teacher is not None, args.use_residual_norm, args.normalize_x, args.normalization_scope,
                     getattr(args, "use_lora", True), sorted(model._lora_years_added),
                     getattr(args, "retrain_match_capacity", False))
    optimizer = model.build_optimizer(args.lr, incremental=teacher is not None)
    best_value, best_path, counter = float("inf"), None, 0
    started = time.perf_counter()
    for epoch in range(args.epoch):
        model.train()
        profile = args.profile_v6_parts and epoch == args.profile_v6_epoch
        args.profile_v6_active = profile
        if profile:
            model.reset_v6_profile()
        if teacher is not None:
            teacher.args.profile_v6_active = False
        training, component_sums = [], {}
        for batch_index, batch in enumerate(train_loader):
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            _check_finite(batch.x, "data.x", args, epoch, batch_index)
            _check_finite(batch.y, "data.y", args, epoch, batch_index)
            pred = model(batch, args.adj)
            _check_finite(pred, "prediction", args, epoch, batch_index)
            loss, parts = compute_loss(pred, batch.y, model, args, batch=batch, teacher_model=teacher)
            _check_finite(loss, "loss", args, epoch, batch_index)
            loss.backward()
            if args.smoke:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        _check_finite(parameter.grad, "gradient " + name, args, epoch, batch_index)
            optimizer.step()
            training.append(float(loss.detach()))
            for name, value in parts.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(value.detach())
        if profile:
            model.log_v6_profile(args.logger, epoch)
        args.profile_v6_active = False
        model.eval()
        validation = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device)
                pred = model(batch, args.adj)
                _check_finite(pred, "validation prediction", args, epoch)
                validation.append(float(masked_mae_np(batch.y.cpu().numpy(), pred.cpu().numpy(), 0)))
        value = float(np.mean(validation))
        components = {name: total / len(training) for name, total in component_sums.items()}
        args.logger.info("task=%s epoch=%s train=%.6f val_mae=%.6f loss_components=%s",
                         args.year, epoch, np.mean(training), value, components)
        _maybe_dump_predictions(model, args, epoch, train_loader, test_loader)
        if value < best_value:
            best_value, counter = value, 0
            best_path = task_dir / f"{value:.8f}.pkl"
            save_checkpoint(model, args, best_path)
        else:
            counter += 1
            if counter > args.patience:
                break
    if best_path is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    saved_path = getattr(args, "checkpoint_path", "")
    try:
        args.checkpoint_path = str(best_path)
        model, _ = load_best_model(args)
    finally:
        args.checkpoint_path = saved_path
    return {"metrics": evaluate(model, args, test_loader), "epochs": epoch + 1,
            "seconds": round(time.perf_counter() - started, 3),
            "best_val_mae": best_value, "checkpoint": best_path.relative_to(Path(args.path)).as_posix()}
