import operator
from functools import partial, reduce

import torch
from torch import nn
from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost
import sys
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.utils import clamp_preserve_gradients
from pyro.distributions.util import copy_docs_from
sys.path.append(".")
from models.nets import ConditionalDenseNN, DenseNN
from utils import expm
from pyro.distributions import constraints

eps = 1e-8

class Exponential_matrix_coupling(TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective=True
    def __init__(self, split_dim, hypernet,input_dim,scale,shift,rescale,reshift, dim=-1,algo='original',eps=1e-8):
        super().__init__(cache_size=1)
        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")
        self.split_dim = split_dim
        self.nn = hypernet
        self.dim = dim
        self.event_dimension = -dim
        self.cached_w_mat = None
        self.input_dim = input_dim
        self.scale = nn.Parameter(scale)
        self.shift = nn.Parameter(shift)
        self.rescale = nn.Parameter(rescale)
        self.reshift = nn.Parameter(reshift)
        self.algo = algo
        self.eps = eps
        



    def _exp(self, x, M):
        """
        Performs power series approximation to the vector product of x with the
        matrix exponential of M.
        """
        power_term = x.unsqueeze(-1)
        y = x.unsqueeze(-1)
        for idx in range(self.iterations):
            power_term = torch.matmul(M, power_term) / (idx + 1)
            y = y + power_term

        return y.squeeze(-1)


    def _trace(self, M):
        """
        Calculates the trace of a matrix and is able to do broadcasting over batch
        dimensions, unlike `torch.trace`.
        Broadcasting is necessary for the conditional version of the transform,
        where `self.weights` may have batch dimensions corresponding the batch
        dimensions of the context variable that was conditioned upon.
        """
        return M.diagonal(dim1=-2, dim2=-1).sum(axis=-1)




    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        # To allow for non batched
        # if len(x.shape) ==2:
        #     x= x.unsqueeze(0)
        x1, x2 = x.split([self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim)

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        w_mat,b_vec = self.nn(x1.reshape(x1.shape[:-self.event_dimension] + (-1,))) 
        w_mat = self.rescale*torch.tanh(self.scale*w_mat+self.shift) +self.reshift + eps
        

        w_mat = w_mat.reshape((w_mat.shape[:-1] + (self.input_dim-self.split_dim,self.input_dim-self.split_dim)))
        

        self.cached_w_mat = w_mat
        w_mat = expm(w_mat,eps = self.eps,algo=self.algo)

       

        y1 = x1
        y2 = torch.matmul(w_mat,x2.unsqueeze(-1)).squeeze(-1) + b_vec

        return torch.cat([y1, y2], dim=self.dim)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
        """
        # To allow for non batched
        # if len(y.shape) ==2:
        #     y= y.unsqueeze(0)
        y1, y2 = y.split([self.split_dim, y.size(self.dim) - self.split_dim], dim=self.dim)
        x1 = y1

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        w_mat,b_vec = self.nn(x1.reshape(x1.shape[:-self.event_dimension] + (-1,)))
        w_mat = self.rescale*torch.tanh(self.scale*w_mat+self.shift) +self.reshift +eps
        w_mat = w_mat.reshape((w_mat.shape[:-1] + (self.input_dim-self.split_dim,self.input_dim-self.split_dim)))
        
        self.cached_w_mat = w_mat
        w_mat = expm(-w_mat,eps = self.eps ,algo=self.algo)
        x2 = torch.matmul(w_mat,(y2-b_vec).unsqueeze(-1)).squeeze(-1)
        return torch.cat([x1, x2], dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """

        if self.cached_w_mat is not None:
            w_mat = self.cached_w_mat
        else:
            x1, _ = x.split([self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim)
            w_mat, _ = self.nn(x1.reshape(x1.shape[:-self.event_dimension] + (-1,)))
            w_mat = self.rescale*torch.tanh(self.scale*w_mat+self.shift) +self.reshift + eps
            w_mat = w_mat.reshape((w_mat.shape[:-1] + (self.input_dim-self.split_dim,self.input_dim-self.split_dim)))
        # Equivalent to : torch.log(torch.abs(torch.det(torch.matrix_exp(w_mat)))) but faster
        return self._trace(w_mat)




class Conditional_exponential_matrix_coupling(ConditionalTransformModule):
    
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, split_dim, hypernet, input_dim,device):
        super().__init__()
        self.split_dim = split_dim
        self.nn = hypernet
        self.input_dim = input_dim
        self.device = device
        self.scale = nn.Parameter(torch.ones(1) / 8).to(device)
        self.shift = nn.Parameter(torch.zeros(1)).to(device)
        self.rescale = nn.Parameter(torch.ones(1)).to(device)
        self.reshift = nn.Parameter(torch.zeros(1)).to(device)

    def condition(self, context):
        cond_nn = partial(self.nn, context=context)
        return Exponential_matrix_coupling(self.split_dim, cond_nn, input_dim = self.input_dim,scale = self.scale,shift= self.shift,rescale = self.rescale,reshift = self.reshift)



def exponential_matrix_coupling(input_dim,device,nonlinearity, hidden_dims=None, split_dim=None, dim=-1):

    if not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError('event shape {} must have same length as event_dim {}'.format(input_dim, -dim))
        event_shape = input_dim
        extra_dims = reduce(operator.mul, event_shape[(dim + 1):], 1)
    else:
        event_shape = [input_dim]
        extra_dims = 1
    event_shape = list(event_shape)

    if split_dim is None:
        split_dim = event_shape[dim] // 2
    if hidden_dims is None:
        hidden_dims = [10 * event_shape[dim] * extra_dims]

    hypernet = DenseNN(split_dim * extra_dims,
                       hidden_dims,
                       param_dims = [(event_shape[dim]-split_dim)**2,event_shape[dim]-split_dim],nonlinearity=nonlinearity)
    scale = nn.Parameter(torch.ones(1) / 8).to(device)
    shift = nn.Parameter(torch.zeros(1)).to(device)
    rescale = nn.Parameter(torch.ones(1)).to(device)
    reshift = nn.Parameter(torch.zeros(1)).to(device)
    return Exponential_matrix_coupling(split_dim, hypernet, dim=dim, input_dim=input_dim,scale = scale,shift= shift,rescale = rescale,reshift = reshift)

def conditional_exponential_matrix_coupling(input_dim, context_dim,device, hidden_dims=None, split_dim=None, dim=-1,nonlinearity=torch.nn.ELU()):
   
    if not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError('event shape {} must have same length as event_dim {}'.format(input_dim, -dim))
        event_shape = input_dim
        extra_dims = reduce(operator.mul, event_shape[(dim + 1):], 1)
    else:
        event_shape = [input_dim]
        extra_dims = 1
    event_shape = list(event_shape)

    if split_dim is None:
        split_dim = event_shape[dim] // 2
    if hidden_dims is None:
        hidden_dims = [10 * event_shape[dim] * extra_dims]
    
    nn = ConditionalDenseNN(input_dim = split_dim * extra_dims,
                            context_dim =  context_dim, 
                            hidden_dims = hidden_dims,
                            param_dims = [(event_shape[dim]-split_dim)**2,event_shape[dim]-split_dim],
                            nonlinearity= nonlinearity
                            )
    return Conditional_exponential_matrix_coupling(split_dim, nn,input_dim = input_dim,device = device)

#Attention Stuff


class ExponentialMatrixCouplngAttn(TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective=True
    def __init__(self, split_dim, hypernet,input_dim, dim=-1,algo='original',eps=1e-8):
        super().__init__(cache_size=1)
        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")
        self.split_dim = split_dim
        self.eps = eps
        self.nn = hypernet
        self.dim = dim
        self.event_dimension = -dim
        self.cached_w_mat = None
        self.input_dim = input_dim
        self.scale = nn.Parameter(torch.ones(1) / 8)
        self.shift = nn.Parameter(torch.zeros(1))
        self.rescale = nn.Parameter(torch.ones(1))
        self.reshift = nn.Parameter(torch.zeros(1))
        self.algo = algo
        self.eps = eps
    

    def _exp(self, x, M):
        """
        Performs power series approximation to the vector product of x with the
        matrix exponential of M.
        """
        power_term = x.unsqueeze(-1)
        y = x.unsqueeze(-1)
        for idx in range(self.iterations):
            power_term = torch.matmul(M, power_term) / (idx + 1)
            y = y + power_term

        return y.squeeze(-1)


    def _trace(self, M):
        """
        Calculates the trace of a matrix and is able to do broadcasting over batch
        dimensions, unlike `torch.trace`.
        Broadcasting is necessary for the conditional version of the transform,
        where `self.weights` may have batch dimensions corresponding the batch
        dimensions of the context variable that was conditioned upon.
        """
        return M.diagonal(dim1=-2, dim2=-1).sum(axis=-1)




    def _call(self, x,attn_emb):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        x1, x2 = x.split([self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim)

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
       
        w_mat,b_vec = self.nn(attn_emb) 
        w_mat = self.rescale*torch.tanh(self.scale*w_mat+self.shift) +self.reshift + eps
        

        w_mat = w_mat.reshape((w_mat.shape[:-1] + (self.input_dim-self.split_dim,self.input_dim-self.split_dim)))
        

        self.cached_w_mat = w_mat
        w_mat = expm(w_mat,eps = self.eps, algo=self.algo)

        

        y1 = x1
        y2 = torch.matmul(w_mat,x2.unsqueeze(-1)).squeeze(-1) + b_vec

        return torch.cat([y1, y2], dim=self.dim)

    def _inverse(self, y,attn_emb):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
        """

        y1, y2 = y.split([self.split_dim, y.size(self.dim) - self.split_dim], dim=self.dim)
        x1 = y1

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
     
        w_mat,b_vec = self.nn(attn_emb) 
        w_mat = self.rescale*torch.tanh(self.scale*w_mat+self.shift) +self.reshift +eps
        w_mat = w_mat.reshape((w_mat.shape[:-1] + (self.input_dim-self.split_dim,self.input_dim-self.split_dim)))
        
        self.cached_w_mat = w_mat
        w_mat = expm(-w_mat,eps = self.eps, algo=self.algo)
        x2 = torch.matmul(w_mat,(y2-b_vec).unsqueeze(-1)).squeeze(-1)
        return torch.cat([x1, x2], dim=self.dim)

    def log_abs_det_jacobian(self,x,y,attn_emb):
        """
        Calculates the elementwise determinant of the log jacobian
        """

        if self.cached_w_mat is not None:
            w_mat = self.cached_w_mat
        else:
            x1, _ = x.split([self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim)
            w_mat, _ = self.nn(attn_emb)
            w_mat = self.rescale*torch.tanh(self.scale*w_mat+self.shift) +self.reshift + eps
            w_mat = w_mat.reshape((w_mat.shape[:-1] + (self.input_dim-self.split_dim,self.input_dim-self.split_dim)))
        # Equivalent to : torch.log(torch.abs(torch.det(torch.matrix_exp(w_mat)))) but faster
        return self._trace(w_mat)









def exponential_matrix_coupling_attn(input_dim,attn_dim,nonlinearity, hidden_dims=None, split_dim=None, dim=-1):
    if not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError('event shape {} must have same length as event_dim {}'.format(input_dim, -dim))
        event_shape = input_dim
        extra_dims = reduce(operator.mul, event_shape[(dim + 1):], 1)
    else:
        event_shape = [input_dim]
        extra_dims = 1
    event_shape = list(event_shape)

    if split_dim is None:
        split_dim = event_shape[dim] // 2
    if hidden_dims is None:
        hidden_dims = [10 * event_shape[dim] * extra_dims]

    hypernet = DenseNN(attn_dim,
                       hidden_dims,
                       param_dims = [(event_shape[dim]-split_dim)**2,event_shape[dim]-split_dim],nonlinearity=nonlinearity)

    return ExponentialMatrixCouplngAttn(split_dim, hypernet, dim=dim, input_dim=input_dim)



if __name__ == '__main__':
    for ind in range(0,100):
        x = torch.randn((32,2000,6))
        context = torch.randn((32,10))
        input_dim = x.shape[-1]
        context_dim = context.shape[-1]
        coupling = conditional_exponential_matrix_coupling(input_dim=input_dim, context_dim=context_dim, hidden_dims=[128,256], split_dim=input_dim//2, dim=-1,device='cpu')
        conditioned_coupling = coupling.condition(context.unsqueeze(-2))
        y = conditioned_coupling._call(x)
        x_after = conditioned_coupling._inverse(y)
        print(torch.abs(x_after-x).max())
        jac = conditioned_coupling.log_abs_det_jacobian(x,y)
        