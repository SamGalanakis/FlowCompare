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
from utils import expm
from models.transform import Transform

eps = 1e-8
class Learned_permuter(TransformModule):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.order_floats = nn.Parameter(torch.randn(dim))
        self.event_dim = -dim

    def _call(self,x):
        permutation_indices = torch.argsort(self.order_floats)
        y = x.index_select(-1,permutation_indices)
        return y
    def _inverse(self,y):
        permutation_indices = torch.argsort(self.order_floats)
        inv_permutation_indices = torch.sort(permutation_indices)[1]
        x = y.index_select(-1,inv_permutation_indices)
        return x
    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(x.size()[:-self.event_dim], dtype=x.dtype, layout=x.layout, device=x.device)


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
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective=True
    def __init__(self,dim,algo='original'):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn((self.dim,self.dim)))
        self.scale = nn.Parameter(torch.ones(1) / 8)
        self.shift = nn.Parameter(torch.zeros(1))
        self.rescale = nn.Parameter(torch.ones(1))
        self.reshift = nn.Parameter(torch.zeros(1))
        self.algo = algo
        
    def _trace(self, M):

        return M.diagonal(dim1=-2, dim2=-1).sum(-1)

    def _inverse(self,y): 
        w_mat = self.rescale*torch.tanh(self.scale*self.w+self.shift) +self.reshift + eps
        return torch.matmul(expm(-w_mat,algo=self.algo),y.unsqueeze(-1)).squeeze(-1)
    def _call(self,x):
        w_mat = self.rescale*torch.tanh(self.scale*self.w+self.shift) +self.reshift + eps
        y = torch.matmul(expm(w_mat,algo=self.algo),x.unsqueeze(-1)).squeeze(-1)
        return y
    def log_abs_det_jacobian(self,x,y):
        w_mat = self.rescale*torch.tanh(self.scale*self.w+self.shift) +self.reshift + eps
        return self._trace(w_mat)
            

class ExponentialCombiner(Transform):
    def __init__(self,dim,algo='original',eps=1e-8,eps_expm=1e-8):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn((self.dim,self.dim)))
        self.scale = nn.Parameter(torch.ones(1) / 8)
        self.shift = nn.Parameter(torch.zeros(1))
        self.rescale = nn.Parameter(torch.ones(1))
        self.reshift = nn.Parameter(torch.zeros(1))
        self.algo = algo
        self.eps = eps
        self.eps_expm = eps_expm
    def _trace(self, M):

        return M.diagonal(dim1=-2, dim2=-1).sum(-1)
    
    def forward(self,x,context=None):
        w_mat = self.rescale*torch.tanh(self.scale*self.w+self.shift) +self.reshift + self.eps
        return torch.matmul(expm(w_mat,eps=self.eps_expm,algo=self.algo),x.unsqueeze(-1)).squeeze(-1), self._trace(w_mat)
    def inverse(self,y,context=None):
        w_mat = self.rescale*torch.tanh(self.scale*self.w+self.shift) +self.reshift + self.eps
        return torch.matmul(expm(-w_mat,eps= self.eps_expm,algo=self.algo),y.unsqueeze(-1)).squeeze(-1)

class Permuter(Transform):
    def __init__(self,permutation,event_dim=-1):
        self.permutation = permutation
        self.event_dim = event_dim
        self.inv_permutation =  torch.sort(self.permutation)[1]


    def forward(self,x,context=None):
        y = x.index_select(self.event_dim,self.permutation)
        return y, torch.zeros(x.size()[:self.event_dim], dtype=x.dtype, layout=x.layout, device=x.device)
    def inverse(self, y ,context=None):
        x = y.index_select(self.event_dim,self.inv_permutation)
        return x










   
if __name__ == '__main__':
    exp_comb = Exponential_combiner(6)
    learned_permuter = Learned_permuter(6)
    for i in range(100):
        a = torch.randn((20,2000,6))
        print(torch.max(torch.abs(a-learned_permuter._inverse(learned_permuter(a)))))

    # for i in tqdm(range(100)):
    #     x = torch.randn((20,6,6))
    #     y = exp_comb(x)
    #     x_ = exp_comb._inverse(y)
    #     print(torch.abs(x-x_).max())
    
