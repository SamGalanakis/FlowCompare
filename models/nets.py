import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self,in_dim,sizes,out_dim,nonlin,residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.sizes = sizes
        self.out_dim = out_dim
        self.nonlin = nonlin
        self.residual = residual
        self.in_layer = nn.Linear(in_dim,self.sizes[0])
        self.out_layer = nn.Linear(self.sizes[-1],out_dim)
        self.layers = nn.ModuleList([nn.Linear(sizes[index],sizes[index+1]) for index in range(len(sizes)-1)])

    def forward(self,x):
        x = self.nonlin(self.in_layer(x))

        for index, layer in enumerate(self.layers):
            if ((index % 2) == 0):
                residual = x
                x = self.nonlin(layer(x))
            else:
                x = self.nonlin(residual+layer(x))

        x = self.out_layer(x)
        return x
