"""Training contract shared by faithful, method-specific baseline adapters.

The common loop controls data splits and evaluation only. Adapters retain their
own incremental updates, training losses, node selection and regularization.
"""

from pathlib import Path
import os
import tempfile
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data import SpatioTemporalDataset
from metrics import MAE_torch, masked_mae_np


def make_loader(inputs, split, args, shuffle=False):
    dataset = SpatioTemporalDataset(inputs, split, steps_per_day=args.steps_per_day)
    if not len(dataset):
        raise ValueError(f"Empty {split} split in task {args.year}")
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=args.device.type == "cuda")


def finite(value, name, args):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{args.baseline}/{args.dataset}/{args.variant}/task={args.year}: non-finite {name}")


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def save_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(cpu_state(payload), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class BaseAdapter:
    """Extend the hooks needed by a particular upstream training algorithm."""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.context = None
        self.train_inputs = None
        self.train_adj = None

    def prepare_task(self, ctx):
        self.context = ctx
        self.train_inputs = ctx.inputs
        self.train_adj = ctx.adj

    def predict(self, batch, training=False):
        return self.model(batch, self.train_adj if training else self.context.adj)

    def training_step(self, batch):
        prediction = self.predict(batch, training=True)
        loss = MAE_torch(prediction, batch.y, 0.0)
        return loss, {"mae": loss}

    def build_optimizer(self):
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("The adapter did not enable any trainable parameters")
        return torch.optim.AdamW(parameters, lr=self.args.lr,
                                 weight_decay=getattr(self.args, "weight_decay", 0.01))

    def before_epoch(self, epoch):
        pass

    def after_task(self):
        pass

    def extra_state(self):
        return {}

    def load_extra_state(self, state):
        pass

    def snapshot(self):
        return {"model_state_dict": cpu_state(self.model.state_dict()),
                "adapter_state": cpu_state(self.extra_state())}

    def restore(self, state):
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.load_extra_state(state.get("adapter_state", {}))


class PrefixMetrics:
    """Accumulate global masked prefix metrics without retaining all predictions."""

    def __init__(self, horizon):
        self.horizon = horizon
        self.count = np.zeros(horizon, dtype=np.int64)
        self.absolute = np.zeros(horizon, dtype=np.float64)
        self.squared = np.zeros(horizon, dtype=np.float64)
        self.percentage = np.zeros(horizon, dtype=np.float64)

    def update(self, truth, prediction):
        truth = np.asarray(truth, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        valid = truth != 0
        difference = prediction - truth
        axes = tuple(range(truth.ndim - 1))
        self.count += valid.sum(axis=axes)
        self.absolute += np.where(valid, np.abs(difference), 0).sum(axis=axes)
        self.squared += np.where(valid, difference ** 2, 0).sum(axis=axes)
        percentage = np.divide(np.abs(difference), np.abs(truth), out=np.zeros_like(truth), where=valid)
        self.percentage += percentage.sum(axis=axes)

    def result(self):
        count = np.maximum(np.cumsum(self.count), 1)
        mae = np.cumsum(self.absolute) / count
        rmse = np.sqrt(np.cumsum(self.squared) / count)
        mape = 100 * np.cumsum(self.percentage) / count
        rows = [{"MAE": float(mae[i]), "RMSE": float(rmse[i]), "MAPE": float(mape[i])}
                for i in range(self.horizon)]
        result = {str(h): rows[h - 1] for h in (3, 6, self.horizon) if h <= self.horizon}
        result["Avg"] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        return result


def validate(adapter, loader, args):
    adapter.model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            prediction = adapter.predict(batch, training=False)
            if prediction.shape != batch.y.shape:
                raise ValueError(f"Validation prediction {prediction.shape} != label {batch.y.shape}")
            finite(prediction, "validation prediction", args)
            values.append(float(masked_mae_np(batch.y.cpu().numpy(), prediction.cpu().numpy(), 0)))
    # Identical checkpoint criterion to the existing PSRAG entry point.
    return float(np.mean(values))


def evaluate(adapter, loader, args):
    adapter.model.eval()
    accumulator = PrefixMetrics(args.y_len)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            prediction = adapter.predict(batch, training=False)
            if prediction.shape != batch.y.shape:
                raise ValueError(f"Test prediction {prediction.shape} != label {batch.y.shape}")
            finite(prediction, "test prediction", args)
            accumulator.update(batch.y.cpu().numpy(), prediction.cpu().numpy())
    result = accumulator.result()
    for horizon, metrics in result.items():
        args.logger.info("test task=%s prefix=%s %s", args.year, horizon, metrics)
    return result


def fit_task(adapter, ctx):
    args = ctx.args
    started = time.perf_counter()
    BaseAdapter.prepare_task(adapter, ctx)
    adapter.prepare_task(ctx)
    adapter.model.to(args.device)
    training = make_loader(adapter.train_inputs, "train", args, shuffle=True)
    validation = make_loader(ctx.inputs, "val", args)
    testing = make_loader(ctx.inputs, "test", args)
    optimizer = adapter.build_optimizer()
    total = sum(p.numel() for p in adapter.model.parameters())
    trainable = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    args.logger.info("task=%s parameters=%s trainable=%s training_nodes=%s samples=%s",
                     args.year, total, trainable, adapter.train_inputs["train_x"].shape[-1],
                     {name: len(ctx.inputs[name + "_x"]) for name in ("train", "val", "test")})
    best_value, best_epoch, counter = float("inf"), -1, 0
    checkpoint = Path(args.path) / str(args.year) / "best.pkl"
    history = []
    for epoch in range(args.epoch):
        adapter.model.train()
        adapter.before_epoch(epoch)
        losses, components = [], {}
        for batch in training:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = adapter.training_step(batch)
            finite(loss, "training loss", args)
            loss.backward()
            if args.smoke:
                for name, parameter in adapter.model.named_parameters():
                    if parameter.grad is not None:
                        finite(parameter.grad, "gradient " + name, args)
            if getattr(args, "grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            for name, value in parts.items():
                components[name] = components.get(name, 0.0) + float(value.detach() if torch.is_tensor(value) else value)
        score = validate(adapter, validation, args)
        if not np.isfinite(score):
            raise FloatingPointError(f"Non-finite validation MAE in task {args.year}")
        parts = {name: value / len(losses) for name, value in components.items()}
        row = {"epoch": epoch, "train": float(np.mean(losses)), "val_mae": score, "loss_components": parts}
        history.append(row)
        args.logger.info("task=%s epoch=%s train=%.6f val_mae=%.6f loss_components=%s",
                         args.year, epoch, row["train"], score, parts)
        if score < best_value:
            best_value, best_epoch, counter = score, epoch, 0
            payload = adapter.snapshot()
            payload["metadata"] = {"format_version": 1, "baseline": args.baseline,
                                   "dataset": args.dataset, "variant": args.variant,
                                   "year": args.year, "nodes": args.graph_size,
                                   "normalize_x": args.normalize_x,
                                   "normalization_scope": args.normalization_scope,
                                   "target_scale": "source", "best_val_mae": score, "epoch": epoch}
            save_atomic(payload, checkpoint)
        else:
            counter += 1
            if counter > args.patience:
                break
    if best_epoch < 0:
        raise RuntimeError("No finite best checkpoint was saved")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapter.restore(saved)
    reloaded_mae = validate(adapter, validation, args)
    if not np.isclose(reloaded_mae, best_value, atol=1e-5, rtol=1e-6):
        raise RuntimeError(f"Best-checkpoint validation mismatch: {best_value} vs {reloaded_mae}")
    metrics = evaluate(adapter, testing, args)
    adapter.after_task()
    # Persist method state updated after selecting the best model (e.g. Fisher).
    completed = adapter.snapshot()
    completed["metadata"] = saved["metadata"]
    save_atomic(completed, checkpoint)
    return {"metrics": metrics, "epochs": epoch + 1, "best_epoch": best_epoch,
            "best_val_mae": best_value, "reloaded_val_mae": reloaded_mae,
            "checkpoint": checkpoint.relative_to(Path(args.path)).as_posix(),
            "parameters": total, "trainable_parameters": trainable,
            "lr": args.lr, "batch_size": args.batch_size,
            "seconds": round(time.perf_counter() - started, 3), "history": history}
