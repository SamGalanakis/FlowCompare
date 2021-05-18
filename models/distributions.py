import torch
import torch.nn as nn
import math
from utils import sum_except_batch,mean_except_batch

#Code adapted from : https://github.com/didriknielsen/survae_flows/



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

    def sample_with_log_prob(self, num_samples,context=None,n_points=None):
        """Generates samples from the distribution together with their log probability.
        Args:
            num_samples: int, number of samples to generate.
        Returns:
            samples: Tensor, shape (num_samples, ...)
            log_prob: Tensor, shape (num_samples,)
        """
        samples = self.sample(num_samples,context=context,n_points=n_points)
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
        return torch.distributions.Normal(loc=mean, scale=self.log_scale.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x),num_dims=2)

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob,num_dims=2)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean



class ConditionalNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1,clamp=False):
        super().__init__()
        self.net = net
        self.clamp=clamp
        

    def cond_dist(self, context):
        
        params = torch.utils.checkpoint.checkpoint(self.net,context,preserve_rng_state=False)
        #params = self.net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=-1)
        scale = log_std.exp()
        if self.clamp:
            scale = scale.clamp_max(self.clamp)
        
        return torch.distributions.Normal(loc=mean, scale=scale)

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x),num_dims=2)

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob,num_dims=2)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev

class StandardUniform(Distribution):
    """A multivariate Uniform with boundaries (0,1)."""

    def __init__(self, shape):
        super().__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('one', torch.ones(1))

    def log_prob(self, x,context=None):
        lb = mean_except_batch(x.ge(self.zero).type(self.zero.dtype),num_dims=2)
        ub = mean_except_batch(x.le(self.one).type(self.one.dtype),num_dims=2)
        return torch.log(lb*ub)

    def sample(self, num_samples,context=None,n_points = None):
        sample_shape = list(self.shape)
        sample_shape[-2] = n_points
        return torch.rand((num_samples,) + sample_shape, device=self.zero.device, dtype=self.zero.dtype)


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super(StandardNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x,context= None):
        log_base =  - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * x**2
        return sum_except_batch(log_base+log_inner,num_dims=2)

    def sample(self, num_samples,context= None,n_points=None):
        sample_shape = list(self.shape)
        sample_shape[-2] = n_points
        return torch.randn(num_samples, *sample_shape, device=self.buffer.device, dtype=self.buffer.dtype)

class Normal(Distribution):
    def __init__ (self,loc,scale,shape):
        super().__init__()
        self.std_normal = StandardNormal(shape)
        self.shape = torch.Size(shape)
        self.register_buffer('loc', loc)
        self.register_buffer('scale', scale)
    def log_prob(self, x,context= None):
        x = (x-self.loc)/self.scale
        return self.std_normal.log_prob(x,context= None)

    def sample(self, num_samples,context= None,n_points=None):
        sample_shape = list(self.shape)
        sample_shape[-2] = n_points
        return (self.std_normal.sample(num_samples,context= None,shape=sample_shape) *self.scale) + self.loc
        
