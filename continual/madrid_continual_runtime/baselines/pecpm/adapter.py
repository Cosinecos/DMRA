"""PECPM network and incremental training with reconstructed preprocessing.

The public repository omits its daily/attention generators. See PROVENANCE.md:
daily averaging, nonoverlapping patterns, warm-start KMeans and top-k cosine
labels follow sections 3.1.2/3.2.2 of the authors' KDD 2023 paper.
"""

import random
from types import SimpleNamespace

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
import torch
from torch.nn import functional as F

from madrid_continual_runtime.common import BaseAdapter, masked_loss
from madrid_continual_runtime.baselines.trafficstream.method import FisherPenalty, node_subgraph, training_daily, observable_seeds
from madrid_continual_runtime.data import cache_metadata
from madrid_continual_runtime.metrics import MAE_torch
from _upstream.pecpm.model import Basic_Model


DEFAULTS = {
    'lr': .01, 'batch_size': 64, 'epoch': 100, 'patience': 10,
    'lr_schedule': [.01, .01, .01, .005, .005, .005, .005],
    'batch_size_schedule': [64, 128, 128, 128, 128, 128, 128],
    'pecpm_hidden': 64, 'pecpm_clusters': 64, 'pecpm_topk': 5,
    'pecpm_num_hops': 2, 'pecpm_evolved_fraction': .01,
    'pecpm_replay_fraction': .09, 'pecpm_stable_replay_fraction': .1,
    'pecpm_cap_trigger': .2, 'pecpm_cap_fraction': .15,
    'pecpm_beta': [5., 5., 1., 1., 0., 1., 1.],
    'pecpm_dropout': [0., .01, 0., 0., 0., 0., .01],
    'pecpm_ewc': True, 'pecpm_ewc_lambda': [.001, .001, .0001, .001, .001, .001, .001],
    'pecpm_fisher_max_batches': 0,
    'fidelity_status': 'adapted_missing_official_preprocessing',
}


def at_year(setting, args):
    if isinstance(setting, (tuple, list)):
        return setting[min(int(args.year - args.begin_year), len(setting) - 1)]
    return setting


class Adapter(BaseAdapter):
    def __init__(self, args):
        super().__init__(args)
        self.ewc = FisherPenalty()
        self.pattern_centers = None
        self.daily_profile = None
        self.mapping = None
        self.subset = None
        self.selected_nodes = None
        self._attention = None
        self._train_mean, self._train_std = 0., 1.

    def prepare_task(self, ctx):
        super().prepare_task(ctx)
        args = self.args
        if not bool(getattr(args, 'init', True)):
            self.model = None
            self.ewc = FisherPenalty()
            self.pattern_centers = None
            self.daily_profile = None
        metadata = cache_metadata(ctx.inputs)
        stats = metadata.get('x_statistics', {}).get('train', {})
        self._train_mean, self._train_std = float(stats.get('mean', 0)), float(stats.get('std', 1))
        self.mapping = None
        self.subset = np.arange(ctx.adj.shape[0])
        self.selected_nodes = self.subset
        previous_profile = self.daily_profile
        madrid = str(args.dataset).startswith('MADRID')
        self.daily_profile = training_daily(ctx.train_raw_data, args.steps_per_day, 1.0,
                                            valid=getattr(ctx, 'train_raw_valid', None) if madrid else None)
        if args.steps_per_day < args.x_len:
            raise ValueError('PECPM daily pattern length must be at least x_len')
        slices = args.steps_per_day // args.x_len
        profile = self.daily_profile
        if str(args.dataset).startswith('MADRID'):
            profile = profile[np.asarray(ctx.active_nodes, dtype=bool)]
        patterns = profile[:, :slices * args.x_len].reshape(-1, args.x_len)
        clusters = int(args.pecpm_clusters)
        if clusters <= 0 or clusters > len(patterns):
            raise ValueError(f'PECPM needs 0 < clusters <= daily patterns ({len(patterns)})')
        if self.pattern_centers is None:
            clustering = KMeans(n_clusters=clusters, random_state=args.seed, n_init=10)
        else:
            # Preserve the semantic identity of bank slots across years.
            clustering = KMeans(n_clusters=clusters, init=self.pattern_centers, n_init=1,
                                random_state=args.seed)
        self.pattern_centers = clustering.fit(patterns).cluster_centers_.astype(np.float32)
        self._pattern_tensor = torch.as_tensor(self.pattern_centers, device=args.device)
        if self.model is None:
            network_args = SimpleNamespace(
                dropout=[float(at_year(args.pecpm_dropout, args))], year=args.year,
                begin_year=args.year, x_len=args.x_len, y_len=args.y_len, cluster=clusters,
                gcn={'in_channel': args.x_len, 'hidden_channel': args.pecpm_hidden, 'out_channel': args.x_len},
                tcn={'in_channel': 1, 'out_channel': 1, 'kernel_size': 3, 'dilation': 1})
            self.model = Basic_Model(network_args).to(args.device)
            args.logger.info('PECPM patterns=%s clusters=%s train_only_daily=True preprocessing_reconstructed=True',
                             len(patterns), clusters)
            return
        self.model.dropout = float(at_year(args.pecpm_dropout, args))
        self.model.eval()
        current_nodes = ctx.adj.shape[0]
        if previous_profile is None:
            previous_profile = training_daily(ctx.previous_raw_data, args.steps_per_day, 1.0,
                                                valid=getattr(ctx, 'previous_raw_valid', None) if madrid else None)
        common = min(len(previous_profile), len(self.daily_profile))
        # Retain the public detect.py's scipy WD on daily values. Its equation
        # differs from paper Eq.7; this discrepancy is explicitly documented.
        distances = np.asarray([wasserstein_distance(previous_profile[node], self.daily_profile[node])
                                for node in range(common)])
        order = np.argsort(distances)
        evolved_count = max(1, int(args.pecpm_evolved_fraction * current_nodes))
        evolved = order[-min(evolved_count, common):].tolist()
        replay_count = max(1, int(args.pecpm_replay_fraction * current_nodes))
        stable_count = max(1, int(replay_count * args.pecpm_stable_replay_fraction))
        stable = order[:min(stable_count, common)].tolist()
        random_count = max(0, replay_count - stable_count)
        replay = np.random.choice(common, random_count, replace=True).tolist() if random_count else []
        new_nodes = list(range(common, current_nodes))
        seeds = sorted(set(new_nodes + evolved + stable + replay))
        if len(seeds) > int(args.pecpm_cap_trigger * current_nodes):
            seeds = sorted(random.sample(seeds, max(1, int(args.pecpm_cap_fraction * current_nodes))))
        seeds = observable_seeds(ctx, seeds)
        self.selected_nodes = np.asarray(seeds)
        if bool(args.pecpm_ewc):
            self.ewc.estimate(self, ctx.inputs, self.prediction_loss, int(args.pecpm_fisher_max_batches))
        self.train_inputs, self.train_adj, self.subset, self.mapping = node_subgraph(
            ctx.inputs, ctx.raw_adj, seeds, args.pecpm_num_hops, args.device)
        self.selected_nodes = self.subset[self.mapping.detach().cpu().numpy()]
        args.logger.info('PECPM new=%s evolved=%s stable=%s random_draws=%s seeds=%s subgraph=%s EWC=%s',
                         len(new_nodes), len(evolved), len(stable), random_count, len(self.selected_nodes), len(self.subset),
                         bool(args.pecpm_ewc))

    def temporal_batch(self, batch, node_count):
        size = batch.x.shape[0] // node_count
        signal = batch.x.reshape(size, node_count, self.args.x_len)
        tod = batch.x_tod.reshape(size, self.args.x_len).to(signal.dtype) / self.args.steps_per_day
        dow = batch.x_dow.reshape(size, self.args.x_len).to(signal.dtype)
        features = torch.stack((signal, tod[:, None].expand_as(signal), dow[:, None].expand_as(signal)), dim=-1)
        combined = batch.clone()
        combined.x = features.reshape(-1, self.args.x_len * 3)
        return combined

    def predict(self, batch, training=False):
        adjacency = self.train_adj if training else self.context.adj
        prepared = self.temporal_batch(batch, adjacency.shape[0])
        prediction, self._attention = self.model(prepared, adjacency)
        return prediction

    def prediction_loss(self, prediction, truth, valid=None):
        # Correct source argument order so the mask depends on labels, not on
        # predictions; all methods use the same zero-valued missing target.
        return masked_loss(prediction, truth, valid, kind='mae', args=self.args)

    def matching_labels(self, batch):
        # The paper matches physical input patterns to physical daily centers.
        # Undo only the DATA TRANSFORM for this auxiliary-label computation;
        # the predictor still receives normalized x and predicts source y.
        original = batch.x * self._train_std + self._train_mean
        normalized_x = F.normalize(original, dim=-1, eps=1e-8)
        normalized_p = F.normalize(self._pattern_tensor, dim=-1, eps=1e-8)
        similarities = normalized_x @ normalized_p.T
        topk = min(int(self.args.pecpm_topk), similarities.shape[-1])
        if topk < 1:
            raise ValueError('pecpm_topk must be positive')
        values, indices = similarities.topk(topk, dim=-1)
        result = torch.zeros_like(similarities)
        result.scatter_(1, indices, values)
        return result

    def training_step(self, batch):
        prediction = self.predict(batch, training=True)
        matching = self.matching_labels(batch)
        if str(self.args.dataset).startswith('MADRID'):
            count = len(self.subset)
            subset = torch.as_tensor(self.subset, device=batch.x.device, dtype=torch.long)
            active = torch.as_tensor(self.context.active_nodes, device=batch.x.device, dtype=torch.bool)[subset]
            keep = active.repeat(batch.x.shape[0] // count)
            if getattr(batch, 'x_valid', None) is not None:
                keep = keep & batch.x_valid.reshape(-1, self.args.x_len).any(-1)
            attention_loss = (F.mse_loss(self._attention[keep], matching[keep]) if keep.any()
                              else self._attention.sum()*0)
        else:
            attention_loss = F.mse_loss(self._attention, matching)
        truth = batch.y
        valid = getattr(batch, 'y_valid', None)
        if self.mapping is not None:
            prediction = prediction.reshape(-1, len(self.subset), self.args.y_len)[:, self.mapping]
            truth = truth.reshape(-1, len(self.subset), self.args.y_len)[:, self.mapping]
            if valid is not None:
                valid = valid.reshape(-1, len(self.subset), self.args.y_len)[:, self.mapping]
        forecast_loss = self.prediction_loss(prediction, truth, valid=valid)
        strength = float(at_year(self.args.pecpm_ewc_lambda, self.args))
        regularization = self.ewc.loss(self.model, strength) if bool(self.args.pecpm_ewc) else forecast_loss.new_zeros(())
        weighted_attention = float(at_year(self.args.pecpm_beta, self.args)) * attention_loss
        return forecast_loss + weighted_attention + regularization, {
            'mae': forecast_loss, 'attention': weighted_attention, 'ewc': regularization}

    def extra_state(self):
        return {'ewc': self.ewc.state(), 'pattern_centers': self.pattern_centers,
                'daily_profile': self.daily_profile, 'selected_nodes': self.selected_nodes,
                'subset': self.subset, 'mapping': self.mapping,
                'train_mean': self._train_mean, 'train_std': self._train_std}

    def load_extra_state(self, state):
        self.ewc.load(state.get('ewc', {}), self.args.device)
        self.pattern_centers = state.get('pattern_centers', self.pattern_centers)
        if self.pattern_centers is not None:
            self._pattern_tensor = torch.as_tensor(self.pattern_centers, device=self.args.device)
        self.daily_profile = state.get('daily_profile', self.daily_profile)
        self.selected_nodes = state.get('selected_nodes', self.selected_nodes)
        self.subset = state.get('subset', self.subset)
        self.mapping = state.get('mapping', self.mapping)
        self._train_mean = state.get('train_mean', self._train_mean)
        self._train_std = state.get('train_std', self._train_std)
        if self.mapping is not None:
            self.mapping = self.mapping.to(self.args.device)
