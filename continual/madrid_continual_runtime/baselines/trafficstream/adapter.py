"""TrafficStream predictor plus native selection and EWC training algorithm."""

import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from madrid_continual_runtime.common import BaseAdapter, masked_loss
from .method import FisherPenalty, feature_js, node_subgraph, recent_features, observable_seeds
from _upstream.trafficstream.model import Basic_Model


DEFAULTS = {
    'lr': .01, 'batch_size': 128, 'epoch': 100, 'patience': 5,
    'lr_schedule': None, 'batch_size_schedule': None,
    'ts_hidden': 64, 'ts_dropout': 0.0, 'ts_num_hops': 2,
    'ts_detect_fraction': .01, 'ts_replay_fraction': .09,
    'ts_max_fraction': .10, 'ts_ewc': True, 'ts_ewc_lambda': .0001,
    'ts_fisher_max_batches': 0, 'ts_prediction_loss': 'mse',
    'fidelity_status': 'official_network_and_incremental_algorithm_train_only_detection',
}


class Adapter(BaseAdapter):
    def __init__(self, args):
        super().__init__(args)
        self.ewc = FisherPenalty()
        self.mapping = None
        self.subset = None
        self.selected_nodes = None

    def prepare_task(self, ctx):
        super().prepare_task(ctx)
        args = self.args
        if not bool(getattr(args, 'init', True)):
            self.model = None
            self.ewc = FisherPenalty()
        self.mapping = None
        self.subset = np.arange(ctx.adj.shape[0])
        self.selected_nodes = self.subset
        if self.model is None:
            network_args = SimpleNamespace(dropout=args.ts_dropout, x_len=args.x_len, y_len=args.y_len,
                gcn={'in_channel': args.x_len, 'hidden_channel': args.ts_hidden, 'out_channel': args.x_len},
                tcn={'in_channel': 1, 'out_channel': 1, 'kernel_size': 3, 'dilation': 1})
            self.model = Basic_Model(network_args).to(args.device)
            return
        self.model.eval()
        previous_nodes = ctx.previous_raw_adj.shape[0]
        current_nodes = ctx.adj.shape[0]
        new_nodes = list(range(previous_nodes, current_nodes))
        previous_adj = np.asarray(ctx.previous_raw_adj, dtype=np.float32)
        previous_adj = torch.as_tensor(previous_adj / (previous_adj.sum(1, keepdims=True) + 1e-6),
                                       device=args.device, dtype=torch.float32)
        old_features = recent_features(self.model, ctx.previous_inputs, previous_adj, args)
        current_features = recent_features(self.model, ctx.inputs, ctx.adj, args)
        drift = feature_js(old_features, current_features)
        evolved_count = max(1, int(args.ts_detect_fraction * current_nodes))
        evolved = np.argsort(drift)[-min(evolved_count, len(drift)):].tolist()
        replay_count = max(1, int(args.ts_replay_fraction * current_nodes))
        replay = np.random.choice(current_nodes, replay_count, replace=True).tolist()
        seeds = sorted(set(new_nodes + evolved + replay))
        limit = max(1, int(args.ts_max_fraction * current_nodes))
        if len(seeds) > limit:
            seeds = sorted(random.sample(seeds, limit))
        seeds = observable_seeds(ctx, seeds)
        self.selected_nodes = np.asarray(seeds)
        if bool(args.ts_ewc):
            self.ewc.estimate(self, ctx.inputs, self.prediction_loss, int(args.ts_fisher_max_batches))
        self.train_inputs, self.train_adj, self.subset, self.mapping = node_subgraph(
            ctx.inputs, ctx.raw_adj, seeds, args.ts_num_hops, args.device)
        self.selected_nodes = self.subset[self.mapping.detach().cpu().numpy()]
        args.logger.info('TrafficStream new=%s evolved=%s replay_draws=%s seeds=%s subgraph=%s EWC=%s',
                         len(new_nodes), len(evolved), replay_count, len(self.selected_nodes), len(self.subset), bool(args.ts_ewc))

    def prediction_loss(self, prediction, truth, valid=None):
        if self.args.ts_prediction_loss not in {'mse', 'huber'}:
            raise ValueError('ts_prediction_loss must be mse or huber')
        return masked_loss(prediction, truth, valid, kind=self.args.ts_prediction_loss, args=self.args)

    def training_step(self, batch):
        prediction = self.predict(batch, training=True)
        truth = batch.y
        valid = getattr(batch, 'y_valid', None)
        if self.mapping is not None:
            prediction = prediction.reshape(-1, len(self.subset), self.args.y_len)[:, self.mapping]
            truth = truth.reshape(-1, len(self.subset), self.args.y_len)[:, self.mapping]
            if valid is not None:
                valid = valid.reshape(-1, len(self.subset), self.args.y_len)[:, self.mapping]
        prediction_loss = self.prediction_loss(prediction, truth, valid=valid)
        regularization = self.ewc.loss(self.model, self.args.ts_ewc_lambda) if bool(self.args.ts_ewc) else prediction_loss.new_zeros(())
        return prediction_loss + regularization, {self.args.ts_prediction_loss: prediction_loss, 'ewc': regularization}

    def extra_state(self):
        return {'ewc': self.ewc.state(), 'selected_nodes': self.selected_nodes,
                'subset': self.subset, 'mapping': self.mapping}

    def load_extra_state(self, state):
        self.ewc.load(state.get('ewc', {}), self.args.device)
        self.selected_nodes = state.get('selected_nodes', self.selected_nodes)
        self.subset = state.get('subset', self.subset)
        self.mapping = state.get('mapping', self.mapping)
        if self.mapping is not None:
            self.mapping = self.mapping.to(self.args.device)
