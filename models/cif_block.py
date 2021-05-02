from models import Augment,Slice, distributions,Transform
import torch
import torch.nn as nn
import 



def CIF_helper(input_dim,augment_dim,distribution,dist_shape,flow):
    transforms = []
    dist  = distribution()
    augmenter = Augment(distribution,input_dim)



class CIFBlockAttn(Transform):
    def __init__(self,input_dim,augment_dim,distribution,dist_shape,flow,attn,pre_attention_mlp,n_flows = 1):
        self.input_dim = input_dim
        self.augment_dim = augment_dim
        self.dist_shape = dist_shape
        self.augmenter =  Augment(distribution,input_dim,split_dim=-1)
        self.slicer = Slice(distribution,input_dim,dim=-1)
        