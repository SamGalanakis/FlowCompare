from torch import nn
import torch
import pyro 
from pyro.distributions.transforms import BatchNorm

class Conditional_flow_layers:
        def __init__(self,flow,n_flow_layers,input_dim,device,permuter,hidden_dims,batchnorm):
            self.transformations = []
            self.n_flow_layers = n_flow_layers
            self.hidden_dims = hidden_dims
            self.batchnorm = batchnorm
            
            for i in range(n_flow_layers):
                flow_module = flow()
                
                
                self.transformations.append(flow_module)
                
                if i<(self.n_flow_layers-1): #Don't add at the end
                    
                    #Batchnorm > permute > flow layer and end with flow layer
                    
          
                    permuter_instance = permuter()
            
                    self.transformations.append(permuter_instance)

                    if self.batchnorm:
                        bn_layer = BatchNorm(input_dim)
                        self.transformations.append(bn_layer)
                    
            self.layer_name_list = [type(x).__name__ for x in self.transformations]
        def make_save_list(self):
            save_list = []
            for x in self.transformations:
                try:
                    save_list.append(x.state_dict())
                    continue
                except:
                    pass
                try:
                    save_list.append(x.permutation)
                    continue
                except:
                    pass
                raise Exception('Can not save object')
            return save_list

        def to(self,device):
            for transform in self.transformations:
                try:
                    transform = transform.to(device)
                except:
                    if isinstance(transform,pyro.distributions.transforms.Permute):
                        transform.permutation=transform.permutation.to(device)
            return self