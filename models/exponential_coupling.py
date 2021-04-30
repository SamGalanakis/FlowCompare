import operator
from functools import partial, reduce
import torch
from torch import nn
from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost
import sys
sys.path.append(".")
from models.nets import ConditionalDenseNN, DenseNN
from utils import expm
from transform import Transform
from .nets import MLP



eps = 1e-8
class ExponentialCoupling(Transform):
    def __init__(self,input_dim,hidden_dims,nonlinearity,context_dim=None,event_dim=-1,algo='original',eps_expm=1e-8):
        super().__init__()
        self.event_dim = event_dim
        self.scale = nn.Parameter(torch.ones(1)/8)
        self.shift = nn.Parameter(torch.zeros(1))
        self.rescale = nn.Parameter(torch.ones(1))
        self.reshift = nn.Parameter(torch.zeros(1))
        self.split_dim = input_dim//2
        self.algo = algo
        self.eps_expm = eps_expm
        out_dim = (self.input_dim - self.split_dim)**2 + self.input_dim - self.split_dim
        self.nn = MLP(input_dim+context_dim,hidden_dims,out_dim,nonlinearity,residual=True)


    def _trace(self, M):
        """
        Calculates the trace of a matrix and is able to do broadcasting over batch
        dimensions, unlike `torch.trace`.
        Broadcasting is necessary for the conditional version of the transform,
        where `self.weights` may have batch dimensions corresponding the batch
        dimensions of the context variable that was conditioned upon.
        """
        return M.diagonal(dim1=-2, dim2=-1).sum(axis=-1)

    def forward(self,x,context):
        x2_size = self.input_dim - self.split_dim
        x1, x2 = x.split([self.split_dim, x2_size], dim=self.event_dim)

        nn_input = torch.cat((x1,context),dim=self.event_dim)
        w_mat,b_vec = self.nn(nn_input).split([x2_size**2,x2_size],dim=-1)
        w_mat = self.rescale*torch.tanh(self.scale*w_mat+self.shift) +self.reshift + eps
        w_mat = w_mat.reshape((w_mat.shape[:-1] + (x2_size,x2_size)))
        w_mat = expm(w_mat,algo=self.algo,eps = self.eps_expm)

        y1 = x1
        y2 = torch.matmul(w_mat,x2.unsqueeze(-1)).squeeze(-1) + b_vec


        return torch.cat([y1, y2], dim=self.event_dim), self._trace(w_mat)
    
    def inverse(self,y,context):
        y2_size = self.input_dim - self.split_dim
        y1, y2 = y.split([self.split_dim, y2_size], dim=self.event_dim)
        x1 = y1


        nn_input = torch.cat((x1,context),dim=self.event_dim)
        w_mat,b_vec = self.nn(nn_input).split([y2_size**2,y2_size],dim=-1)
        w_mat = self.rescale*torch.tanh(self.scale*w_mat+self.shift) +self.reshift + eps
        w_mat = w_mat.reshape((w_mat.shape[:-1] + (x2_size,x2_size)))
        #Negative w_mat for inv
        w_mat = expm(-w_mat,algo=self.algo,eps = self.eps_expm)

        
        x2 = torch.matmul(w_mat,(y2-b_vec).unsqueeze(-1)).squeeze(-1)

        return torch.cat([x1, x2], dim=self.dim)

if __name__ == '__main__':
    model = ExponentialCoupling(6,[34, 85],torch.nn.ReLU(),24)

    x = torch.randn((50,2000,6))
    cond = torch.randn((50,2000,24))

    y,ldj = model.forward(x, cond)
    inv = model.inverse(y, cond)
    pass

    