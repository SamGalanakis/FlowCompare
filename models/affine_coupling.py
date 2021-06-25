
import torch
from torch import nn
from models.nets import MLP
from models import Transform
import einops


class AffineCoupling(Transform):
    def __init__(self, input_dim, hidden_dims, nonlinearity, context_dim=0, event_dim=-1, scale_fn_type='exp', eps=1E-8,split_dim=None):
        super().__init__()
        self.event_dim = event_dim
        self.input_dim = input_dim
        if split_dim ==  None:
            self.split_dim = input_dim//2
        else: 
            self.split_dim = split_dim
        self.context_dim = context_dim
        out_dim = (self.input_dim - self.split_dim)*2
        self.nn = MLP(self.split_dim + context_dim, hidden_dims,
                      out_dim, nonlinearity, residual=True)
        self.scale_fn_type = scale_fn_type
        if self.scale_fn_type == 'exp':
            self.scale_fn = lambda x: torch.exp(x)
        elif self.scale_fn_type == 'sigmoid':
            self.scale_fn = lambda x: (2*torch.sigmoid(x) - 1) * (1 - eps) + 1
        else:
            raise Exception('Invalid scale_fn_type')

    def forward(self, x, context=None):
        x2_size = self.input_dim - self.split_dim
        x1, x2 = x.split([self.split_dim, x2_size], dim=self.event_dim)

        nn_input = torch.cat(
            (x1, context), dim=self.event_dim) if self.context_dim != 0 else x1

        s, t = torch.utils.checkpoint.checkpoint(
            self.nn, nn_input, preserve_rng_state=False).split([x2_size, x2_size], dim=-1)

        s = self.scale_fn(s)

        y1 = x1
        y2 = x2*s + t

        ldj = torch.einsum('bnd->bn', torch.log(s))
        return torch.cat([y1, y2], dim=self.event_dim), ldj

    def inverse(self, y, context=None):
        y2_size = self.input_dim - self.split_dim
        y1, y2 = y.split([self.split_dim, y2_size], dim=self.event_dim)
        x1 = y1

        nn_input = torch.cat(
            (y1, context), dim=self.event_dim) if self.context_dim != 0 else y1

        s, t = self.nn(nn_input).split([y2_size, y2_size], dim=-1)

        s = self.scale_fn(s)

        x2 = (y2-t)/s

        return torch.cat([x1, x2], dim=self.event_dim)
