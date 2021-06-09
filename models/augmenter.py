import torch
from models import Transform
from .distributions import ConditionalDistribution

# Code adapted from : https://github.com/didriknielsen/survae_flows/


class Augment(Transform):
    '''
    A simple augmentation layer which augments the input with additional elements.
    This is useful for constructing augmented normalizing flows [1, 2].
    References:
        [1] Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models,
            Huang et al., 2020, https://arxiv.org/abs/2002.07101
        [2] VFlow: More Expressive Generative Flows with Variational Data Augmentation,
            Chen et al., 2020, https://arxiv.org/abs/2002.09741
    '''

    def __init__(self, noise_dist, x_size, split_dim=1):
        super().__init__()
        self.noise_dist = noise_dist
        self.split_dim = split_dim
        self.x_size = x_size
        self.cond = isinstance(self.noise_dist, ConditionalDistribution)

    def split_z(self, z):
        split_proportions = (
            self.x_size, z.shape[self.split_dim] - self.x_size)
        return torch.split(z, split_proportions, dim=self.split_dim)

    def forward(self, x, context=None):
        if context is not None and self.cond:
            context = torch.cat((x, context), axis=self.split_dim)
        else:
            context = x
        if self.cond:
            z2, logqz2 = self.noise_dist.sample_with_log_prob(context=context)
        else:
            #TODO FIX HACK BELOW FOR POINTWISE
            if len(x.shape) == 2:
                num_samples=1
            else:
                num_samples=x.shape[0]
            z2, logqz2 = self.noise_dist.sample_with_log_prob(
                num_samples=num_samples, n_points=x.shape[-2])
        if len(x.shape) == 2:
            z2, logqz2 = z2.squeeze(),logqz2.squeeze()
        z = torch.cat([x, z2], dim=self.split_dim)
        ldj = -logqz2
        return z, ldj

    def inverse(self, z, context=None):
        x, z2 = self.split_z(z)
        return x
