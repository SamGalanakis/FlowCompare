import torch
import torch.nn as nn
import math
from torch.distributions import Normal

#Code adapted from : https://github.com/didriknielsen/survae_flows/

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_except_batch(x, num_dims=1):
    '''
    Averages all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_mean: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).mean(-1)


class Distribution(nn.Module):
    """Distribution base class."""

    def log_prob(self, x,context=None):
        """Calculate log probability under the distribution.
        Args:
            x: Tensor, shape (batch_size, ...)
        Returns:
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def sample(self, num_samples,context=None):
        """Generates samples from the distribution.
        Args:
            num_samples: int, number of samples to generate.
        Returns:
            samples: Tensor, shape (num_samples, ...)
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, num_samples,context=None):
        """Generates samples from the distribution together with their log probability.
        Args:
            num_samples: int, number of samples to generate.
        Returns:
            samples: Tensor, shape (num_samples, ...)
            log_prob: Tensor, shape (num_samples,)
        """
        samples = self.sample(num_samples,context=context)
        log_prob = self.log_prob(samples,context=context)
        return samples, log_prob

    def forward(self, *args, mode, **kwargs):
        '''
        To allow Distribution objects to be wrapped by DataParallelDistribution,
        which parallelizes .forward() of replicas on subsets of data.
        DataParallelDistribution.log_prob() calls DataParallel.forward().
        DataParallel.forward() calls Distribution.forward() for different
        data subsets on each device and returns the combined outputs.
        '''
        if mode == 'log_prob':
            return self.log_prob(*args, **kwargs)
        else:
            raise RuntimeError("Mode {} not supported.".format(mode))
    
class ConditionalDistribution(Distribution):
    """ConditionalDistribution base class"""

    def log_prob(self, x, context):
        """Calculate log probability under the distribution.
        Args:
            x: Tensor, shape (batch_size, ...).
            context: Tensor, shape (batch_size, ...).
        Returns:
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def sample(self, context):
        """Generates samples from the distribution.
        Args:
            context: Tensor, shape (batch_size, ...).
        Returns:
            samples: Tensor, shape (batch_size, ...).
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, context):
        """Generates samples from the distribution together with their log probability.
        Args:
            context: Tensor, shape (batch_size, ...).
        Returns::
            samples: Tensor, shape (batch_size, ...).
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

class ConditionalMeanStdNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and learned std."""

    def __init__(self, net, scale_shape):
        super(ConditionalMeanStdNormal, self).__init__()
        self.net = net
        self.log_scale = nn.Parameter(torch.zeros(scale_shape))

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.log_scale.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean

    
class StandardUniform(Distribution):
    """A multivariate Uniform with boundaries (0,1)."""

    def __init__(self, shape):
        super().__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('one', torch.ones(1))

    def log_prob(self, x,context=None):
        lb = mean_except_batch(x.ge(self.zero).type(self.zero.dtype))
        ub = mean_except_batch(x.le(self.one).type(self.one.dtype))
        return torch.log(lb*ub)

    def sample(self, num_samples,context=None):
        return torch.rand((num_samples,) + self.shape, device=self.zero.device, dtype=self.zero.dtype)


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super(StandardNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x,context= None):
        log_base =  - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * x**2
        return sum_except_batch(log_base+log_inner)

    def sample(self, num_samples,context= None):
        return torch.randn(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype)