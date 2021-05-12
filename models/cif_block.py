from models import transform
from models.transform import Transform
from models import Augment,Slice, PreConditionApplier, IdentityTransform, ActNormBijectionCloud, Reverse
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

def cif_helper(input_dim,augment_dim,distribution,context_dim,flow,attn,pre_attention_mlp,event_dim,conditional_aug,conditional_slice):
    #CIF if aug>base latent dim else normal flow
    if  input_dim < augment_dim:
        return CIFblock(input_dim,augment_dim,distribution,context_dim,flow,attn,pre_attention_mlp,event_dim=event_dim,conditional_aug=conditional_aug,conditional_slice=conditional_slice)
    elif input_dim == augment_dim:
        
        return PreConditionApplier(flow(input_dim,context_dim),CouplingPreconditionerAttn(attn(),pre_attention_mlp(input_dim//2),input_dim//2,event_dim=event_dim))
    else:
        raise Exception('Augment dim smaller than main latent!')


class CIFblock(Transform):
    def __init__(self,input_dim,augment_dim,distribution,context_dim,flow,attn,pre_attention_mlp,event_dim,conditional_aug=True,conditional_slice=True):
        super().__init__()
        self.input_dim = input_dim
        self.augment_dim = augment_dim
        self.event_dim = event_dim
        self.conditional_aug = conditional_aug
        self.conditional_slice = conditional_slice
        
 
        distrib_augment = distribution()
        distrib_slice = distribution()
        self.augmenter =  Augment(distrib_augment,input_dim,split_dim=event_dim)
        
        pre_attention_mlp_input_dim =  augment_dim-input_dim # - input_dim//2  #Context is x1,noise
        self.attention = attn()
        self.pre_attention_mlp =  pre_attention_mlp(pre_attention_mlp_input_dim)
        self.flow = flow(input_dim=augment_dim,context_dim = context_dim)
        self.slicer = Slice(distrib_slice,input_dim,dim=self.event_dim)
        self.reverse = Reverse(augment_dim,dim=self.event_dim)

        if self.conditional_aug:
            self.pre_attention_mlp_aug = pre_attention_mlp(input_dim)
            self.attention_aug = attn()
        
        if self.conditional_slice:
            self.pre_attention_mlp_slice = pre_attention_mlp(input_dim)
            self.attention_slice = attn()


        
    def forward(self,x,context=None):
        ldj_cif = torch.zeros(x.shape[:-1], device=x.device,dtype=x.dtype)

        context_aug = self.attention_aug(self.pre_attention_mlp_aug(x)) if self.conditional_aug else None
        combined,ldj = self.augmenter(x,context=context_aug)
        ldj_cif +=ldj 
        x,noise = combined.split([self.input_dim,self.augment_dim-self.input_dim],dim=self.event_dim)
        
        attention_emb = self.attention(self.pre_attention_mlp(noise),context=context)
        
        x,_ = self.reverse(combined,context=None) #0 ldj
        x,ldj = self.flow(x,context=attention_emb)
        ldj_cif+=ldj
        
        x,_ = self.reverse(x,context=None)

        context_slice = self.attention_slice(self.pre_attention_mlp_slice(x[...,:self.input_dim])) if self.conditional_slice else None
        x,ldj =self.slicer(x,context=context_slice)
        ldj_cif+= ldj

        return x,ldj_cif
    def inverse(self,y,context=None):

        context_slice = self.attention_slice(self.pre_attention_mlp_slice(y)) if self.conditional_slice else None
        y = self.slicer.inverse(y,context=context_slice)
        y = self.reverse.inverse(y,context=None)

        noise,_ = torch.split(y,[self.augment_dim-self.input_dim,self.input_dim],dim=self.event_dim)
        attention_emb = self.attention(self.pre_attention_mlp(noise),context=context)

        y = self.flow.inverse(y,context=attention_emb)
        y = self.reverse.inverse(y,context=None)

        
        y = self.augmenter.inverse(y,context = None)
            
        return y
        

        
        


    
