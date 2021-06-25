import torch
from models import Transform
from .distributions import ConditionalDistribution

# Code adapted from : https://github.com/didriknielsen/survae_flows/

class AugmentAttentionPreconditioner(Transform):
    '''Wraps Augmenter to apply attn before passing context in forward'''
    def __init__(self,augment,attn,pre_attn_mlp):
        super().__init__()
        self.augment = augment 
        self.attn = attn()
        self.pre_attn_mlp = pre_attn_mlp

    def forward(self,x,context):
        attention_emb = self.attn(self.pre_attn_mlp(x),context)
        return self.augment(x,attention_emb)
    def inverse(self,z,context=None):
        return self.augment.inverse(z,context=None)



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

    def __init__(self, noise_dist, x_size, split_dim=1,use_context=True):
        super().__init__()
        self.noise_dist = noise_dist
        self.split_dim = split_dim
        self.x_size = x_size
        self.cond = isinstance(self.noise_dist, ConditionalDistribution)
        self.use_context = use_context

    def split_z(self, z):
        split_proportions = (
            self.x_size, z.shape[self.split_dim] - self.x_size)
        return torch.split(z, split_proportions, dim=self.split_dim)

    def forward(self, x, context=None):
        if context is not None and self.cond and self.use_context:
            context = torch.cat((x, context), axis=self.split_dim)
        else:
            context = x
        if self.cond:
            z2, logqz2 = self.noise_dist.sample_with_log_prob(context=context)
        else:
            
            z2, logqz2 = self.noise_dist.sample_with_log_prob(
                num_samples=x.shape[0], n_points=x.shape[-2])
   
        z = torch.cat([x, z2], dim=self.split_dim)
        ldj = -logqz2
        return z, ldj

    def inverse(self, z, context=None):
        x, z2 = self.split_z(z)
        return x
