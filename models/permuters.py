import torch
from torch.distributions.transforms import Transform
from pyro.distributions.torch_transform import TransformModule
from torch import nn
from torch.nn import functional as F
from pyro.nn.dense_nn import DenseNN
from pyro.distributions.conditional import ConditionalTransformModule
#from pyro.distributions.transforms.matrix_exponential  import conditional_matrix_exponential 
import math
from functools import partial
from torch.distributions import Transform, constraints
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.nn import DenseNN   
from tqdm import tqdm

eps = 1e-8

class Full_matrix_combiner(TransformModule):
    def __init__(self,dim):
        super().__init__()
        self.bijective = True
        self.lin_layer = nn.Linear(dim,dim,bias=False)
     
        
    def _inverse(self,y):    
        return torch.matmul(y,torch.inverse(self.lin_layer.weight / torch.max(torch.abs(self.lin_layer.weight))))
    def _call(self,y):
        return F.linear(y, self.lin_layer.weight / torch.max(torch.abs(self.lin_layer.weight)), self.lin_layer.bias)
    def log_abs_det_jacobian(self,x,y):
        return torch.log(torch.abs(torch.det(torch.inverse(self.lin_layer.weight / torch.max(torch.abs(self.lin_layer.weight)))))) #Jacobian=Matrix for lin maps

class Exponential_combiner(TransformModule):
    def __init__(self,dim,iterations=8):
        super().__init__()
        self.dim = dim
        self.iterations = iterations
        self.w = nn.Parameter(torch.randn((self.dim,self.dim)))
        self.scale = nn.Parameter(torch.ones(1) / 8)
        self.shift = nn.Parameter(torch.zeros(1))
        self.rescale = nn.Parameter(torch.ones(1))
        self.reshift = nn.Parameter(torch.zeros(1))
        



    def _matrix_exp(self,x,M):
        power_term = x.unsqueeze(-1)
        y = x.unsqueeze(-1)
        for idx in range(self.iterations):
            power_term = torch.matmul(M, power_term) / (idx + 1)
            y = y + power_term

        return y.squeeze(-1)
      

    def _trace(self, M):

        return M.diagonal(dim1=-2, dim2=-1).sum(-1)

    def _inverse(self,y): 
        w_mat = self.rescale*torch.tanh(self.scale*self.w+self.shift) +self.reshift + eps
        return torch.matmul(torch.matrix_exp(-w_mat),y.unsqueeze(-1)).squeeze(-1)
    def _call(self,x):
        w_mat = self.rescale*torch.tanh(self.scale*self.w+self.shift) +self.reshift + eps
        y = torch.matmul(torch.matrix_exp(w_mat),x.unsqueeze(-1)).squeeze(-1)
        return y
    def log_abs_det_jacobian(self,x,y):
        w_mat = self.rescale*torch.tanh(self.scale*self.w+self.shift) +self.reshift + eps
        return self._trace(w_mat)
            







   
if __name__ == '__main__':
    exp_comb = Exponential_combiner(6)


    for i in tqdm(range(100)):
        x = torch.randn((20,6,6))
        y = exp_comb(x)
        x_ = exp_comb._inverse(y)
        print(torch.abs(x-x_).max())
    
