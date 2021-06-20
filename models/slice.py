import torch
from models import Transform, ConditionalDistribution

# Code adapted from : https://github.com/didriknielsen/survae_flows/


class Slice(Transform):
    '''
    A simple slice layer which factors out some elements and returns
    the remaining elements for further transformation.
    This is useful for constructing multi-scale architectures [1].
    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    '''

    stochastic_forward = False

    def __init__(self, noise_dist, num_keep, dim=1):
        super().__init__()
        self.noise_dist = noise_dist
        self.dim = dim
        self.num_keep = num_keep
        self.cond = isinstance(self.noise_dist, ConditionalDistribution)

    def split_input(self, input):
        split_proportions = (
            self.num_keep, input.shape[self.dim] - self.num_keep)
        return torch.split(input, split_proportions, dim=self.dim)

    def forward(self, x, context=None):

        z, x2 = self.split_input(x)

        if context is not None:
            context = torch.cat((z, context), axis=self.dim)
        else:
            context = z

        if self.cond:
            ldj = self.noise_dist.log_prob(x2, context=context)
        else:
            ldj = self.noise_dist.log_prob(x2)
        return z, ldj

    def inverse(self, z, context=None):

        if context is not None:
            context = torch.cat((z, context), axis=self.dim)
        else:
            context = z

        if self.cond:
            x2 = self.noise_dist.sample(context=context)
        else:
            x2 = self.noise_dist.sample(num_samples=z.shape[0],n_points = z.shape[1])
        x = torch.cat([z, x2], dim=self.dim)
        return x
