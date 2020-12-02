import torch
import torch.nn.functional as F
from torch import nn
from models.nets import ResBlock, activation_func_selector


class CouplingLayer(nn.Module):
    def __init__(self,coupling_function,split_index,permute_list):
        super().__init__()
        self.coupling_function  = coupling_function
        self.split_index = split_index
        self.permute_list = permute_list

       

    def forward_x(self,x,e = None):
        #if e != None then conditional coupling layer
        x1 = x[:,:self.split_index]
        x2 = x[:,self.split_index:]
        
        if e == None:
            y2 = self.coupling_function(x1,x2)
        else:
            y2 = self.coupling_function(x1,x2,e)
        y = torch.cat((x1,y2),dim=1)
        #Permute 
        y = torch.index_select(y,1,torch.LongTensor([self.permute_list]))
        return y
    
    def inverse(self,y,e = None):
        #Un-permute 
        y = torch.index_select(y,1,torch.LongTensor([self.permute_list]))
        x1 = y1 = y[:,self.split_index]
        y2 = y[:,self.split_index]
        if e==None:
            x2 = self.coupling_function.inverse(y1,y2)
        else: 
            x2 = self.coupling_function.inverse(y1,y2,e)

        x = torch.cat((x1,x2),dim=1)
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
       
        return y2
    def inverse(self,y1,y2,e=None):
        if e == None:
            A = self.add_func(x1)
            M = self.multiply_func(x1)
        else:
            A = self.add_func(x1,e)
            M = self.multiply_func(x1,e)
        x1 = y1
        x2 = (y2 - A )   / torch.exp(M)
       
        return x2 






class ConditionalNet(nn.Module):
    def __init__(self,emb_dim,in_dim,in_dim_total = 3,n_neurons=512,n_cond_pre=2,n_in_pre=1,n_joint=4,base_block_type = 'resnet',activation='leaky_relu'):
        super().__init__()

        self.emb_dim = emb_dim #emb dim of e (conditional)
        self.in_dim = in_dim # dimension of the x (x1) not 3 since x is split  
        self.in_dim_total = in_dim_total # Dimension of pre split x so can calculate dimension of output
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
        self.joint_final_layer = nn.Sequential(nn.Linear(self.n_neurons,self.in_dim_total-self.in_dim),nn.Tanh())



    def forward(self,x,e):
        # x is the split input and e is the conditioner
        e = self.cond_initial_layer(e)
        e = self.in_pre_layers(e)

        x = self.in_initial_layer(x)
        x = self.in_pre_layers(x)

        
        x_joint = torch.cat((x,e),dim=1)


        x_joint = self.joint_layers(x_joint)

        x_joint = self.joint_final_layer(x_joint)

        return x_joint


class StraightNet(nn.Module):
    def __init__(self,input_dim,n_neurons=512,n_blocks=4,base_block_type = 'resnet',activation='leaky_relu'):
        super().__init__()
        self.input_dim = input_dim #in and out are same dim
        self.n_neurons = n_neurons
        self.n_blocks = n_blocks
        self.base_block_type = base_block_type
        self.activation = activation
        self.activation_func = activation_func_selector(self.activation)

        if  self.base_block_type == 'resnet':
            self.base_block = ResBlock

       
        self.in_layer = nn.Sequential(nn.Linear(self.input_dim,self.n_neurons),nn.Tanh())
        self.middle_layers = nn.Sequential(*[self.base_block(n_neurons,2,self.activation)]*self.n_blocks)
        self.final_layer =  nn.Sequential(nn.Linear(self.n_neurons, self.input_dim),nn.Tanh())
        
    def forward(self,x):
        x = self.in_layer(x)
        x= self.middle_layers(x)
        x= self.final_layer(x)
        return x


# class FBlock(nn.Module):
#     def __init__(self,coupling_layer,split_index_list,permute_list_list):
#         super().__init__()
#         self.coupling_layer = self.coupling_layer
#         self.split_index_list = split_index_list
#         self.permute_list_list = permute_list_list
#         coupling_layer_params = zip()
#         self.layers = [self.coupling_layer(coupling_function,split_index,permute_list) for coupling_function,split_index,permute_list]
    
#     def forward(self,x,e):
#         for index in range(len(self.permute_list_list)):
#             permute_list = self.permute_list_list[index]
#             split_index = self.split_index_list[index]
#             x = coupling_


    

        

if __name__ == '__main__':
    F = ConditionalNet(32,2)
    G = StraightNet(32)
    print(F)
    print(G)
    x = torch.randn(2).reshape(1,-1)
    e  = torch.randn(32).reshape(1,-1)
    result_straight = G(e)
    result_cond  =  F(x,e)
    





            





