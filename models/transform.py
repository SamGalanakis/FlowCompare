import torch

import torch.nn as nn

#Code adapted from : https://github.com/didriknielsen/survae_flows/

class Transform(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        
    def forward(self, x,context=None):
        """
        Forward transform.
        Computes `z <- x` and the log-likelihood contribution term `log C`
        such that `log p(x) = log p(z) + log C`.
        Args:
            x: Tensor, shape (batch_size, ...)
        Returns:
            z: Tensor, shape (batch_size, ...)
            ldj: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def inverse(self, z,context=None):
        """
        Inverse transform.
        Computes `x <- z`.
        Args:
            z: Tensor, shape (batch_size, ...)
        Returns:
            x: Tensor, shape (batch_size, ...)
        """
        raise NotImplementedError()



class PreConditionApplier(Transform):
    def __init__(self,transform,pre_conditioner):
        super().__init__()
        self.pre_conditioner = pre_conditioner
        
        self.transform = transform

    def forward(self,x,context=None):
        context_for_transform= self.pre_conditioner(x,context)
        x,ldj = self.transform(x,context = context_for_transform)
        return x,ldj
    def inverse(self,y,context):
        context_for_transform= self.pre_conditioner(y,context)
        y= self.transform.inverse(y,context = context_for_transform)
        return y


class Flow(Transform):
    '''Wrapper for merging multiple transforms'''

    def __init__(self,transform_list,base_dist,sample_dist=None):
        super().__init__()
        self.base_dist = base_dist
        self.sample_dist = sample_dist if sample_dist!=None else base_dist
        self.transforms = nn.ModuleList(transform_list)

    def log_prob(self, x,context=None):
        log_prob = torch.zeros(x.shape[:-1], device=x.device,dtype=x.dtype)
        for index,transform in enumerate(self.transforms):
            x, ldj = transform(x,context=context)
            log_prob += ldj
        log_prob += self.base_dist.log_prob(x)
        return log_prob

    
    def sample(self,num_samples,n_points,context=None,sample_distrib=None):
        dist_for_sample = sample_distrib if sample_distrib!= None else self.sample_dist
        z = dist_for_sample.sample(num_samples,n_points=n_points)
        for transform in reversed(self.transforms):
            z = transform.inverse(z,context=context)
        return z

class IdentityTransform(Transform):
    def __init__(self):
        super().__init__()
    def forward(self,x,context=None):
        return x, 0
    def inverse(self,y,context=None):
        return y
