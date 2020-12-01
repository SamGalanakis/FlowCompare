import torch
import torch.nn.functional as F
from torch import nn

def activation_func_selector(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.1, inplace=True)],
        ['none', nn.Identity()]
    ])[activation]



class ResBlock(nn.Module):
    def __init__(self,n_neurons,n_layers=2,activation='leaky_relu'):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.activation=activation
        self.activation_func = activation_func_selector('leaky_relu')

        self.linear_layers = [nn.Linear(n_neurons,n_neurons) for x in range(n_layers)]
        
        
    def forward(self,x):
        x_skip = x
        for lin_layer in self.linear_layers[:-1]:
            x= lin_layer(x)
            self.activation_func(x)
          
        x=self.linear_layers[-1](x)
        x+=x_skip
        self.activation_func(x)
        return x