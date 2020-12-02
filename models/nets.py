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
    def __init__(self,n_neurons,n_layers=1,activation='leaky_relu'):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.activation=activation
        self.activation_func = activation_func_selector('leaky_relu')

        component_list = [nn.Linear(n_neurons,n_neurons),self.activation_func]*(n_layers-1)
        component_list.append(nn.Linear(n_neurons,n_neurons))
        self.linear_layers = nn.Sequential(*component_list)
        
        
    def forward(self,x):
        x_skip = x
        x = self.linear_layers(x)
        x+=x_skip
        x = self.activation_func(x)
        return x


if __name__ == '__main__':
    a = ResBlock(5,2)
    b = torch.randn(5)
    a(b)
    print(a)
