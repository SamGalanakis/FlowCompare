from models import Transform
import torch
import torch.nn as nn



class Flow(Transform):
    '''Wrapper for merging multiple transforms'''

    def __init__(self,transform_list):
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