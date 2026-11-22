"""Official Graph WaveNet with project history/calendar/graph adapters."""

# https://github.com/nnzhan/Graph-WaveNet/tree/6b162e80c59a1d494809252eca055cff93dc66b1
import numpy as np
import torch
from torch import nn

from .vendor.model import gwnet


NATIVE_LOSS = 'mae'
DEFAULTS = {
    'seq_len': 12, 'pred_len': 12,
    'residual_channels': 32, 'dilation_channels': 32,
    'skip_channels': 256, 'end_channels': 512,
    'kernel_size': 2, 'blocks': 4, 'layers': 2, 'dropout': .3,
    'gcn_bool': True, 'addaptadj': True, 'use_time_of_day': True,
}


def row_normalize(adjacency):
    """Match official util.asym_adj: zero-degree rows remain zero."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    sums = adjacency.sum(axis=1)
    inverse = np.divide(np.ones_like(sums), sums,
                        out=np.zeros_like(sums), where=sums != 0)
    return adjacency * inverse[:, None]


class Forecaster(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown GWNet model settings: {sorted(unknown)}')
        self.config = dict(DEFAULTS, **config)
        self.num_nodes = int(context['num_nodes'])
        self.steps_per_day = int(context['steps_per_day'])
        if self.num_nodes <= 0 or self.steps_per_day <= 0:
            raise ValueError('GWNet needs positive num_nodes and steps_per_day')
        for name in ('seq_len', 'pred_len', 'residual_channels', 'dilation_channels',
                     'skip_channels', 'end_channels', 'kernel_size', 'blocks', 'layers'):
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'GWNet {name} must be a positive integer')
        if self.config['kernel_size'] < 2:
            raise ValueError('GWNet kernel_size must be at least2')
        for name in ('gcn_bool', 'addaptadj', 'use_time_of_day'):
            if not isinstance(self.config[name], bool):
                raise ValueError(f'GWNet {name} must be a boolean')
        dropout = self.config['dropout']
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError('GWNet dropout must be in [0,1)')
        if 'raw_adj' not in context:
            raise ValueError('GWNet requires the prepared physical graph raw_adj')
        adjacency = context['raw_adj']
        if isinstance(adjacency, torch.Tensor):
            adjacency = adjacency.detach().cpu().numpy()
        adjacency = np.asarray(adjacency, dtype=np.float32)
        if (adjacency.shape != (self.num_nodes, self.num_nodes)
                or not np.isfinite(adjacency).all() or (adjacency < 0).any()):
            raise ValueError('GWNet raw_adj must be a finite nonnegative N-by-N matrix')
        self.register_buffer('physical_forward', torch.from_numpy(row_normalize(adjacency).copy()))
        self.register_buffer('physical_backward', torch.from_numpy(row_normalize(adjacency.T).copy()))
        # Construct on CPU; the trainer subsequently moves all parameters and
        # buffers together. The original supports list itself is not a buffer.
        self.core = gwnet(
            device=torch.device('cpu'), num_nodes=self.num_nodes,
            dropout=dropout, supports=[self.physical_forward, self.physical_backward],
            gcn_bool=self.config['gcn_bool'], addaptadj=self.config['addaptadj'],
            aptinit=None, in_dim=2 if self.config['use_time_of_day'] else 1,
            out_dim=self.config['pred_len'],
            **{key: self.config[key] for key in ('residual_channels', 'dilation_channels',
                'skip_channels', 'end_channels', 'kernel_size', 'blocks', 'layers')})

    def forward(self, x, x_tod=None, x_dow=None, y_tod=None, y_dow=None):
        if x.ndim != 3 or x.shape[1:] != (self.num_nodes, self.config['seq_len']):
            raise ValueError(f"GWNet expects [B,{self.num_nodes},{self.config['seq_len']}], got {tuple(x.shape)}")
        history = x.unsqueeze(1)
        if self.config['use_time_of_day']:
            if x_tod is None or x_tod.shape != (x.shape[0], self.config['seq_len']):
                raise ValueError('GWNet historical TOD indices must have shape[B,seq_len]')
            if not torch.isfinite(x_tod).all() or ((x_tod < 0) | (x_tod >= self.steps_per_day)).any():
                raise ValueError('GWNet historical TOD indices must lie in[0,steps_per_day)')
            tod = x_tod.to(device=x.device, dtype=x.dtype) / self.steps_per_day
            tod = tod[:, None, None, :].expand(-1, 1, self.num_nodes, -1)
            history = torch.cat((history, tod), dim=1)
        self.core.supports = [self.physical_forward, self.physical_backward]
        prediction = self.core(history)
        # Default RF13 and history12 produce a single temporal location.
        # For configurable shallower stacks, use the most recent location.
        return prediction[..., -1].transpose(1, 2).contiguous()


def build_model(config, context):
    return Forecaster(config, context)
