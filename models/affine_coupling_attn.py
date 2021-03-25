import torch
from torch.distributions.utils import _sum_rightmost
#from pyro.nn import ConditionalDenseNN, DenseNN
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.utils import clamp_preserve_gradients
from pyro.distributions import constraints
from functools import partial, reduce
import operator
from models.nets import ConditionalDenseNN, DenseNN
class AffineCouplingAttn(TransformModule):
   

    bijective = True

    def __init__(self, split_dim, hypernet, *, dim=-1, log_scale_min_clip=-5., log_scale_max_clip=3.):
        super().__init__(cache_size=1)
        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")

        self.split_dim = split_dim
        self.nn = hypernet
        self.dim = dim
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(constraints.real, -self.dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(constraints.real, -self.dim)

    def _call(self, x,attn_emb):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        x1, x2 = x.split([self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim)

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        mean, log_scale = self.nn(attn_emb)
        mean = mean.reshape(mean.shape[:-1] + x2.shape[self.dim:])
        log_scale = log_scale.reshape(log_scale.shape[:-1] + x2.shape[self.dim:])

        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale

        y1 = x1
        y2 = torch.exp(log_scale) * x2 + mean
        return torch.cat([y1, y2], dim=self.dim)

    def _inverse(self, y,attn_emb):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
        """
        y1, y2 = y.split([self.split_dim, y.size(self.dim) - self.split_dim], dim=self.dim)
        x1 = y1

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        mean, log_scale = self.nn(attn_emb)
        mean = mean.reshape(mean.shape[:-1] + y2.shape[self.dim:])
        log_scale = log_scale.reshape(log_scale.shape[:-1] + y2.shape[self.dim:])

        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale

        x2 = (y2 - mean) * torch.exp(-log_scale)
        return torch.cat([x1, x2], dim=self.dim)

    def log_abs_det_jacobian(self, x, y,attn_emb):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        x_old, y_old = self._cached_x_y
        if self._cached_log_scale is not None and x is x_old and y is y_old:
            log_scale = self._cached_log_scale
        else:
            x1, x2 = x.split([self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim)
            _, log_scale = self.nn(attn_emb)
            log_scale = log_scale.reshape(log_scale.shape[:-1] + x2.shape[self.dim:])
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        return _sum_rightmost(log_scale, self.event_dim)


def affine_coupling_attn(input_dim, attn_dim, hidden_dims=None, split_dim=None, dim=-1, **kwargs):
 
    
    if not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError('event shape {} must have same length as event_dim {}'.format(input_dim, -dim))
        event_shape = input_dim
        extra_dims = reduce(operator.mul, event_shape[(dim + 1):], 1)
    else:
        event_shape = [input_dim]
        extra_dims = 1
    event_shape = list(event_shape)

    if split_dim is None:
        split_dim = event_shape[dim] // 2
    if hidden_dims is None:
        hidden_dims = [10 * event_shape[dim] * extra_dims]

    hypernet = DenseNN(attn_dim,
                       hidden_dims,
                       [(event_shape[dim] - split_dim) * extra_dims,
                        (event_shape[dim] - split_dim) * extra_dims])
    return AffineCouplingAttn(split_dim, hypernet, dim=dim, **kwargs)