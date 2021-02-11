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
        y = self._matrix_exp(y, -self.w)
        return y
    def _call(self,x):
        y = self._matrix_exp(x, self.w)
        return y
    def log_abs_det_jacobian(self,x,y):
        return self._trace(self.w)
            







   
if __name__ == '__main__':
    layer = Linear_combiner(6)
    x = torch.randn(6)
    
