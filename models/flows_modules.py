import torch
import torch.nn.functional as F
from torch import nn



class CouplingLayer(nn.Module):
    def __init__(self,coupling_function,split_index,permute_list):
        super().__init__()
        self.coupling_function  = coupling_function
        self.split_index = split_index
        self.permute_list = permute_list

       

    def forward_x(self,x):
        
        x1 = x[:,:self.split_index]
        x2 = x[:,self.split_index:]
        
        y2 = self.coupling_function(x1,x2)
        y = torch.cat((x1,y2),dim=1)
        #Permute 
        y = torch.index_select(y,1,torch.LongTensor([self.permute_list]))
        return y
    
    def inverse(self,y):
        #Un-permute 
        y = torch.index_select(y,1,torch.LongTensor([self.permute_list]))
        x1 = y1 = y[:,self.split_index]
        y2 = y[:,self.split_index]
        x2 = self.coupling_function.inverse(y1,y2)

        x = torch.cat((x1,x2),dim=1)
        return x


class AffineCouplingFunc(nn.Module):
    def __init__(self,mutiply_func,add_func):
        self.multiply_func = mutiply_func
        self.add_func = add_func

    def forward(self,x1,x2):
        A = self.add_func(x1)
        M = self.multiply_func(x1)
        y2 = x2 * torch.exp(M) + A
        ldet = 
        return y2
    def inverse(self,y1,y2):
        A = self.add_func(x1)
        M = self.multiply_func(x1)
        x1 = y1
        x2 = (y2 - A )   / torch.exp(M))
        ldet = M.sum()
        return x2 , ldet 






class MultiplyNet(torch.nn):
    def __init__(self,conditional,in_dim,emb_dim,n_neurons):
        self.conditional = conditional 
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.n_neurons = n_neurons
        







            





