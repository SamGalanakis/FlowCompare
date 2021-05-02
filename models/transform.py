import torch

import torch.nn as nn


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
        self.module = transform

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

    def __init__(self,transform_list):
        super().__init__()
        self.transforms = nn.ModuleList(transform_list)

    def forward(self,x,context=None):
        ldj_final=0.0
        for transform in self.transforms:
            x,ldj = transform(x,context)
            ldj_final +=ldj
        return x,ldj
    def inverse(self,y,context=None):
        for transform in reversed(self.transforms):
            y = transform.inverse(y,context)
        return y