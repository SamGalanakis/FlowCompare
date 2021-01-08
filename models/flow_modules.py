import torch
import torch.nn.functional as F
from torch import nn
from models.nets import ResBlock, activation_func_selector
import torch.autograd.profiler as profiler

class CouplingLayer(nn.Module):
    def __init__(self,coupling_function,split_index,permute_tensor):
        super().__init__()
        self.coupling_function  = coupling_function
        self.split_index = split_index
        
        self.permute_tensor = permute_tensor.squeeze()

        self.inv_permute_tensor = torch.LongTensor([self.permute_tensor.tolist().index(x) for x in range(len(self.permute_tensor))]).squeeze()
        self.inv_permute_tensor = self.inv_permute_tensor.to(self.permute_tensor.device) 
       

    def forward(self,x,e = None):
        #if e != None then conditional coupling layer
        #split LAST dimension according to split index
        with profiler.record_function("INITIAL SPLIT"):
            x1 = x[...,:self.split_index]
            x2 = x[...,self.split_index:]
        
        if e == None:
            y2, ldetJ = self.coupling_function(x1,x2)
        else:
            y2 , ldetJ = self.coupling_function(x1,x2,e)
        #change this back ? the dim
        with profiler.record_function("CONCAT"):
            y = torch.cat((x1,y2),dim=-1)
        #Permute 
        with profiler.record_function("Index Select"):
            y = torch.index_select(y,-1,self.permute_tensor)
        return y , ldetJ
    
    def inverse(self,y,e = None):
        #Un-permute 
        y = torch.index_select(y,-1,self.inv_permute_tensor)
        x1 = y1 = y[...,:self.split_index]
        y2 = y[...,self.split_index:]
        if e==None:
            x2 = self.coupling_function.inverse(y1,y2)
        else: 
            x2 = self.coupling_function.inverse(y1,y2,e)

        x = torch.cat((x1,x2),dim=-1)
        return x


class AffineCouplingFunc(nn.Module):
    def __init__(self,mutiply_func,add_func):
        super().__init__()
        self.multiply_func = mutiply_func
        self.add_func = add_func

    def forward(self,x1,x2,e=None):
        #if e != None then conditional coupling function
        if e == None:
            A = self.add_func(x1)
            M = self.multiply_func(x1)
        else:
            A = self.add_func(x1,e)
            M = self.multiply_func(x1,e)
        y2 = x2 * torch.exp(M) + A
        ldetJ = torch.sum(M, dim=-1)
        return y2, ldetJ
    def inverse(self,y1,y2,e=None):
        if e == None:
            A = self.add_func(y1)
            M = self.multiply_func(y1)
        else:
            A = self.add_func(y1,e)
            M = self.multiply_func(y1,e)
        x1 = y1
        x2 = (y2 - A )   / torch.exp(M)

        return x2 






class ConditionalNet(nn.Module):
    def __init__(self,emb_dim,in_dim,out_dim,n_neurons=512,n_cond_pre=2,n_in_pre=1,n_joint=4,base_block_type = 'resnet',activation='leaky_relu'):
        super().__init__()

        self.emb_dim = emb_dim #emb dim of e (conditional)
        self.in_dim = in_dim # dimension of the x (x1) not 3 since x is split  
        self.out_dim = out_dim # Dimension of pre split x so can calculate dimension of output
        self.n_neurons = n_neurons
        self.n_cond_pre = n_cond_pre
        self.n_in_pre = n_in_pre
        self.n_joint = n_joint
        self.base_block_type = base_block_type
        self.activation = activation
        self.activation_func = activation_func_selector(self.activation)

        if  self.base_block_type == 'resnet':
            self.base_block = ResBlock
        #Inital layers take input size accordingly and then feed it into layers of premade block, n_neurons is halved since this is 
        # before the conditional and input are concatonated

        self.cond_initial_layer = nn.Sequential(nn.Linear(self.emb_dim,self.n_neurons//2),self.activation_func)
        self.cond_pre_layers = nn.Sequential(*[self.base_block(self.n_neurons//2,2,self.activation)]*self.n_cond_pre)

        self.in_initial_layer = nn.Sequential(nn.Linear(self.in_dim,self.n_neurons//2),self.activation_func)
        self.in_pre_layers = nn.Sequential(*[self.base_block(self.n_neurons//2,2,self.activation)]*self.n_in_pre)


        self.joint_layers = nn.Sequential(*[self.base_block(n_neurons,2,self.activation)]*self.n_joint)
        # Output should be of dimension x2 so dim(x) - dim(x1)
        self.joint_final_layer = nn.Sequential(nn.Linear(self.n_neurons,self.out_dim),nn.Tanh())



    def forward(self,x,e):
        # x is the split input and e is the conditioner
        e = self.cond_initial_layer(e)
        e = self.in_pre_layers(e)

        x = self.in_initial_layer(x)
        x = self.in_pre_layers(x)


        # change dim back?
        x_joint = torch.cat((x,e),dim=-1)

        
        x_joint = self.joint_layers(x_joint)

        x_joint = self.joint_final_layer(x_joint)

        return x_joint


class StraightNet(nn.Module):
    def __init__(self,in_dim,out_dim,n_neurons=512,n_blocks=4,base_block_type = 'resnet',activation='leaky_relu'):
        super().__init__()
        self.in_dim = in_dim 
        self.out_dim  = out_dim
        self.n_neurons = n_neurons
        self.n_blocks = n_blocks
        self.base_block_type = base_block_type
        self.activation = activation
        self.activation_func = activation_func_selector(self.activation)

        if  self.base_block_type == 'resnet':
            self.base_block = ResBlock

       
        self.in_layer = nn.Sequential(nn.Linear(self.in_dim,self.n_neurons),nn.Tanh())
        self.middle_layers = nn.Sequential(*[self.base_block(n_neurons,2,self.activation)]*self.n_blocks)
        self.final_layer =  nn.Sequential(nn.Linear(self.n_neurons, self.out_dim),nn.Tanh())
        
    def forward(self,x):
        x = self.in_layer(x)
        x= self.middle_layers(x)
        x= self.final_layer(x)
        return x




    

        

if __name__ == '__main__':
    F = ConditionalNet(32,2)
    G = StraightNet(32)
    print(F)
    print(G)
    x = torch.randn(2).reshape(1,-1)
    e  = torch.randn(32).reshape(1,-1)
    result_straight = G(e)
    result_cond  =  F(x,e)
    





            





