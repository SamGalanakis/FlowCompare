import torch
from torch.distributions.transforms import Transform
from pyro.distributions.torch_transform import TransformModule
from torch import nn
from torch.nn import functional as F
from pyro.nn.dense_nn import DenseNN
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.transforms.matrix_exponential  import conditional_matrix_exponential 
    
    
def scale(x,inital_bound,new_bound):
    initial_bound_length = inital_bound[1] - inital_bound[0]
    bound_length_new = new_bound[1] - new_bound[0]

    return (x+(-inital_bound[0]+new_bound[0]))*( bound_length_new/initial_bound_length )

    

class Sigmoid_scaler(TransformModule):
    def __init__(self):
        super().__init__()
        self.bijective = True

     
        
    def _inverse(self,y):    
        return (torch.sigmoid(y)*6) -3
    def _call(self,x):
        return torch.logit((x+3)/6)
    def log_abs_det_jacobian(self,x,y):
        derivative = -6/(-9 + x**2)
        if (derivative>100000).any():
            print()
        return torch.abs(derivative)
    
class Tanh_scaler(Transform):
    def __init__(self):
        super().init()
        self.bijective=True
    def _call(self,x):
        F.tanh

    def _inverse(self,y):
        self.last_max = y.max()
        self.last_min = y.min()
        y= (y-self.last_min)/self.last_max
        return y
    def log_abs_det_jacobian(self,x,y):
        derivative = -6/(-9 + x**2)
        if (derivative>100000).any():
            print()
        return torch.abs(derivative)

if __name__ == "__main__":
    scaler = Sigmoid_scaler()
    pass










