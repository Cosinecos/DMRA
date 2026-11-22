"""Train each stage and pass its validation-best state to the next stage."""
import gc
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import implementation
from .continual import CHECKPOINT_FORMAT, inherit_best, state_sha256


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _forward(model, batch):
    result = model(batch['x'], batch['x_tod'], batch['x_dow'], batch['y_tod'], batch['y_dow'])
    if result.shape != batch['y'].shape:
        raise ValueError(f'Prediction shape {tuple(result.shape)} != label {tuple(batch["y"].shape)}')
    if not torch.isfinite(result).all():
        raise FloatingPointError('Non-finite forecast; check model/learning rate/input scale')
    return result


def chunks(batch, size, device):
    for start in range(0, len(batch['x']), size):
        yield {key: value[start:start + size].to(device, non_blocking=True) for key, value in batch.items()}


def training_valid(batch, task, loss_kind):
    valid = batch['valid'].bool() & torch.isfinite(batch['y'])
    if not task.madrid and loss_kind == 'mae':
        valid = valid & (batch['y'].abs() > 5e-5)
    return valid


def loss_sum(prediction, truth, valid, kind):
    difference = prediction - truth
    values = difference.abs() if kind == 'mae' else difference.square()
    return torch.where(valid, values, torch.zeros_like(values)).sum()


class PrefixMetrics:
    """Global sums preserve true Madrid zeros and legacy prefix definitions."""
    def __init__(self, madrid):
        self.madrid = madrid
        self.sums = np.zeros((7, 12), dtype=np.float64)

    def update(self, pred, truth, valid):
        pred, truth = pred.detach().double(), truth.detach().double()
        valid = valid.bool() & torch.isfinite(truth)
        error = pred - truth
        nonzero = valid & (truth > 0 if self.madrid else truth != 0)
        axes = (0, 1)
        rows = [valid.sum(axes), torch.where(valid, error.abs(), 0).sum(axes),
                torch.where(valid, error.square(), 0).sum(axes),
                torch.where(valid, truth.abs(), 0).sum(axes),
                torch.where(nonzero, error.abs() / truth.abs().clamp_min(1e-12), 0).sum(axes),
                nonzero.sum(axes), (valid & (truth == 0)).sum(axes)]
        self.sums += torch.stack(rows).cpu().numpy()

    def result(self):
        sums = self.sums.cumsum(1)
        prefixes = []
        result = {}
        for index in range(12):
            count = sums[0, index]
            if count == 0:
                raise ValueError('No valid validation/test targets')
            row = {'MAE': float(sums[1, index] / count),
                   'RMSE': float(np.sqrt(sums[2, index] / count)),
                   'MAPE': float(100 * sums[4, index] / sums[5, index]) if sums[5, index] else None}
            if self.madrid:
                row.update(WAPE=float(100 * sums[1, index] / sums[3, index]) if sums[3, index] else None,
                           observed_targets=int(count), zero_targets=int(sums[6, index]),
                           MAPE_targets=int(sums[5, index]))
            prefixes.append(row)
            if index + 1 in (3, 6, 12):
                result[str(index + 1)] = row
        keys = ('MAE', 'RMSE', 'MAPE', 'WAPE') if self.madrid else ('MAE', 'RMSE', 'MAPE')
        result['Avg'] = {key: float(np.mean([row[key] for row in prefixes if row[key] is not None]))
                         if any(row[key] is not None for row in prefixes) else None for key in keys}
        if self.madrid:
            result['Avg'].update({key: prefixes[-1][key] for key in ('observed_targets', 'zero_targets', 'MAPE_targets')})
        return result


def evaluate(model, loader, task, config, device, validation=False):
    model.eval()
    accumulator = PrefixMetrics(task.madrid)
    batch_scores = []
    with torch.no_grad():
        for batch in loader:
            count, absolute = 0, 0.
            for part in chunks(batch, config['microbatch_size'], device):
                pred = _forward(model, part)
                accumulator.update(pred, part['y'], part['valid'])
                valid = part['valid'].bool()
                count += int(valid.sum())
                absolute += float(torch.where(valid, (pred - part['y']).abs(), 0).double().sum())
            if count:
                batch_scores.append(absolute / count)
            elif not task.madrid:
                # Legacy masked_mae_np returns zero for an all-zero batch;
                # preserve its contribution to the effective-batch mean.
                batch_scores.append(0.)
    metrics = accumulator.result()
    if validation:
        # Madrid uses globally observed MAE. The other datasets use the mean
        # of each effective batch's masked MAE to select the best epoch.
        return metrics['12']['MAE'] if task.madrid else float(np.mean(batch_scores))
    return metrics


def fit_task(task, config, checkpoint, logger, device, previous_checkpoint):
    """Train one stage; full_ft transfers weights, never the optimizer or graph."""
    if (task.year > config['begin_year']) != (previous_checkpoint is not None):
        raise ValueError('A preceding validation-best checkpoint is required after the first stage')
    seed_everything(config['seed'])
    module = implementation(config['model_name'])
    model = module.build_model(config['model'], task.context()).to(device)
    fresh_initial_sha256 = state_sha256(model.state_dict())
    initialization_report = None
    if previous_checkpoint is not None:
        initialization_report = inherit_best(model, task, config, previous_checkpoint, device)
    # Fixed attention scales in upstream PatchTST are intentionally not learned.
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    initial_sha256 = state_sha256(model.state_dict())
    point_loss = module.NATIVE_LOSS
    initialized_from = task.year - 1 if previous_checkpoint is not None else None
    native_fixed = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    logger.info('task=%s nodes=%s parameters=%s trainable=%s strategy=full_ft initialized_from=%s '
                'initial_state_sha256=%s loss=%s batch_size=%s microbatch_size=%s preprocessing=%s',
                task.label, task.nodes, total, trainable, initialized_from, initial_sha256, point_loss,
                config['batch_size'], config['microbatch_size'], task.metadata)
    logger.info('task=%s optimizer_initialized_from=none native_fixed_parameters=%s initialization_report=%s',
                task.label, native_fixed, initialization_report)
    loaders = {split: DataLoader(task.dataset(split), batch_size=config['batch_size'], shuffle=split == 'train',
                                num_workers=config['num_workers'], pin_memory=device.type == 'cuda')
               for split in ('train', 'val', 'test')}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config['lr'], weight_decay=config['weight_decay'])
    best_value, best_epoch, counter = float('inf'), -1, 0
    history = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    training_seconds = validation_seconds = checkpoint_seconds = 0.
    for epoch in range(config['epoch']):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss, epoch_batches = 0., 0
        skipped = 0
        for batch_index, batch in enumerate(loaders['train']):
            denominator = int(training_valid(batch, task, point_loss).sum())
            if denominator == 0:
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for part in chunks(batch, config['microbatch_size'], device):
                valid = training_valid(part, task, point_loss)
                if not valid.any():
                    continue
                pred = _forward(model, part)
                primary = loss_sum(pred, part['y'], valid, point_loss) / denominator
                batch_loss += float(primary.detach())
                if not torch.isfinite(primary):
                    raise FloatingPointError('Non-finite training loss')
                primary.backward()
            if config['smoke']:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f'Non-finite gradient: {name}')
            if config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            epoch_batches += 1
            epoch_loss += batch_loss
            if config['log_every'] > 0 and (batch_index + 1) % config['log_every'] == 0:
                logger.info('task=%s epoch=%s batch=%s/%s loss=%.6f', task.label, epoch,
                            batch_index + 1, len(loaders['train']), batch_loss)
        if not epoch_batches:
            raise ValueError('No valid training labels in this task')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed_train = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        score = evaluate(model, loaders['val'], task, config, device, validation=True)
        if not np.isfinite(score):
            raise FloatingPointError('Non-finite validation MAE')
        elapsed_val = time.perf_counter() - validation_start
        total_loss = epoch_loss / epoch_batches
        row = {'epoch': epoch, 'train': total_loss, 'val_mae': score,
               'train_seconds': elapsed_train, 'validation_seconds': elapsed_val, 'skipped_unobserved_batches': skipped}
        history.append(row)
        training_seconds += elapsed_train
        validation_seconds += elapsed_val
        logger.info('task=%s epoch=%s train=%.6f val_mae=%.6f train_seconds=%.3f val_seconds=%.3f',
                    task.label, epoch, total_loss, score, elapsed_train, elapsed_val)
        if score < best_value:
            best_value, best_epoch, counter = score, epoch, 0
            metadata = {'format': CHECKPOINT_FORMAT, 'model': config['model_name'],
                        'dataset': config['dataset'], 'variant': config['variant'], 'year': task.year,
                        'num_nodes': task.nodes, 'strategy': 'full_ft', 'node_ids': task.node_ids.tolist(),
                        'config': config, 'preprocessing': task.metadata}
            save_start = time.perf_counter()
            save_atomic({'state_dict': cpu_state(model), 'metadata': metadata}, checkpoint)
            checkpoint_seconds += time.perf_counter() - save_start
        else:
            counter += 1
            if counter > config['patience']:
                break
    before = next(iter(loaders['val']))
    probe = {key: value[:min(config['microbatch_size'], 2)].to(device) for key, value in before.items()}
    model.eval()
    # Compare after restoring the checkpoint, not after the last training epoch.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'], strict=True)
    with torch.no_grad():
        expected = _forward(model, probe).detach().cpu()
    selected_hash = state_sha256(model.state_dict())
    del optimizer, model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Reconstruct from config/context and strict-load every buffer and parameter.
    seed_everything(config['seed'])
    model = module.build_model(config['model'], task.context()).to(device)
    if state_sha256(model.state_dict()) != fresh_initial_sha256:
        raise RuntimeError('Fresh initialization is not reproducible for this task')
    model.load_state_dict(saved['state_dict'], strict=True)
    if state_sha256(model.state_dict()) != selected_hash:
        raise RuntimeError('Strict checkpoint state differs after reconstruction')
    model.eval()
    with torch.no_grad():
        actual = _forward(model, probe).detach().cpu()
    reload_error = float((expected - actual).abs().max())
    if reload_error > 1e-6:
        raise RuntimeError(f'Checkpoint prediction mismatch: {reload_error}')
    reloaded_score = evaluate(model, loaders['val'], task, config, device, validation=True)
    if not np.isclose(reloaded_score, best_value, atol=1e-5, rtol=1e-6):
        raise RuntimeError(f'Reloaded validation mismatch: {best_value} vs {reloaded_score}')
    metrics = evaluate(model, loaders['test'], task, config, device)
    for prefix, values in metrics.items():
        logger.info('test task=%s prefix=%s %s', task.label, prefix, values)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None
    result = {'metrics': metrics, 'epochs': len(history), 'best_epoch': best_epoch, 'best_val_mae': best_value,
              'reloaded_val_mae': reloaded_score, 'checkpoint_prediction_max_abs_error': reload_error,
              'initial_state_sha256': initial_sha256, 'selected_state_sha256': selected_hash,
              'fresh_initial_state_sha256': fresh_initial_sha256,
              'strategy': 'full_ft', 'initialized_from': initialized_from, 'optimizer_initialized_from': None,
              'initialization_report': initialization_report, 'native_fixed_parameters': native_fixed,
              'parameters': total, 'trainable_parameters': trainable, 'loss': point_loss,
              'source_sha256': task.source_sha256, 'graph_sha256': task.graph_sha256,
              'seconds': time.perf_counter() - started, 'training_seconds': training_seconds,
              'validation_seconds': validation_seconds, 'checkpoint_seconds': checkpoint_seconds,
              'peak_cuda_allocated_bytes': peak_memory, 'history': history, 'preprocessing': task.metadata}
    del model, saved, loaders, before, probe, expected, actual
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result
