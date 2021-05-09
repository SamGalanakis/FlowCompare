from models import transform
from models.transform import Transform
from models import Augment,Slice, PreConditionApplier, IdentityTransform, ActNormBijectionCloud,Reverse
import torch
import torch.nn as nn 





class CouplingPreconditionerAttn(nn.Module):
    def __init__(self,attn,pre_attention_mlp,x1_dim,event_dim=-1):
        super().__init__()
        self.attn = attn
        self.pre_attention_mlp = pre_attention_mlp
        self.event_dim = event_dim
        self.x1_dim = x1_dim
    def forward(self,x,context):
        x1,x2 = x.split([self.x1_dim, self.x1_dim], dim=self.event_dim)
        attn_emb = self.attn(self.pre_attention_mlp(x1),context = context)
        return attn_emb

def cif_helper(input_dim,augment_dim,distribution,context_dim,flow,attn,pre_attention_mlp,n_flows,event_dim):
    #CIF if aug>base latent dim else normal flow
    if  input_dim < augment_dim:
        return CIFblock(input_dim,augment_dim,distribution,context_dim,flow,attn,pre_attention_mlp,event_dim=-1)
    elif input_dim == augment_dim:
        PreConditionApplier(flow(input_dim,input_dim),CouplingPreconditionerAttn(attn(),pre_attention_mlp,input_dim//2,event_dim=event_dim))
        cif_block = lambda: PreConditionApplier(flow(input_dim,input_dim),CouplingPreconditionerAttn(attn(),pre_attention_mlp,input_dim//2,event_dim=event_dim))
    else:
        raise Exception('Augment dim smaller than main latent!')


class CIFblock(Transform):
    def __init__(self,input_dim,augment_dim,distribution,context_dim,flow,attn,pre_attention_mlp,event_dim):
        super().__init__()
        self.input_dim = input_dim
        self.augment_dim = augment_dim
        self.event_dim = event_dim
        
        
 
        distrib_augment = distribution()
        distrib_slice = distribution()
        self.augmenter =  Augment(distrib_augment,input_dim,split_dim=event_dim)
        
        pre_attention_mlp_input_dim =  augment_dim-input_dim # - input_dim//2  #Context is x1,noise
        self.attention = attn()
        self.pre_attention_mlp =  pre_attention_mlp(pre_attention_mlp_input_dim)
        self.flow = flow(input_dim=augment_dim,context_dim = context_dim)
        self.slicer = Slice(distrib_slice,input_dim,dim=self.event_dim)
        self.reverse = Reverse(augment_dim,dim=self.event_dim)

        
    def forward(self,x,context=None):
        ldj_cif = torch.zeros(x.shape[:-1], device=x.device,dtype=x.dtype)
        combined,ldj = self.augmenter(x,context=None)
        ldj_cif +=ldj 
        x,noise = combined.split([self.input_dim,self.augment_dim-self.input_dim],dim=self.event_dim)
        
        attention_emb = self.attention(self.pre_attention_mlp(noise),context=context)
        
        x,_ = self.reverse(combined,context=None) #0 ldj
        x,ldj = self.flow(x,context=attention_emb)
        ldj_cif+=ldj
        
        x._ = self.reverse(x,context=None)
        x,ldj =self.slicer(x,context=None)
        ldj_cif+= ldj

        return x,ldj_cif
    def inverse(self,y,context=None):
        y = self.slicer.inverse(y,context=None)
        y = self.reverse.inverse(y,context=None)

        noise,_ = torch.split(y,[self.augment_dim-self.input_dim,self.input_dim],dim=self.event_dim)
        attention_emb = self.attention(self.pre_attention_mlp(noise),context=context)

        y = self.flow.inverse(y,context=attention_emb)
        y = self.reverse.inverse(y,context=None)

        y = self.augmenter.inverse(y,context =None)
            
        return y
        

        
        


    
