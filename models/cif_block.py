from models import Augment,Slice, distributions,Transform, Flow, PreConditionApplier
import torch
import torch.nn as nn 






class CouplingPreconditionerAttn(nn.Module):
    def __init__(self,attn,pre_attention_mlp,split_dim,event_dim=-1):
        super().__init__()
        self.attn = attn
        self.pre_attn_mlp = pre_attention_mlp
        self.event_dim = event_dim
        self.split_dim = split_dim
    def forward(self,x,context):
        x1,x2 = x.split([self.split_dim, x.size(self.event_dim) - self.split_dim], dim=self.event_dim)
        attn_emb = self.attn(self.pre_attention_mlp(x1),context = context)
        return attn_emb

def get_cif_block_attn(input_dim,augment_dim,distribution,context_dim,sample_size,flow,attn,pre_attention_mlp,n_flows,split_dim,event_dim):
    transforms = []
    distrib_augment = distribution(shape = (sample_size,augment_dim-input_dim))
    distrib_slice = distribution(shape = (sample_size,augment_dim-input_dim))
    augmenter =  Augment(distrib_augment,input_dim,split_dim=-1)
    transforms.append(augmenter)
    
    for x in range(n_flows):
        temp_flow = flow(input_dim=augment_dim,context_dim = context_dim)
        temp_coupling_preconditioner_attn = CouplingPreconditionerAttn(attn(),pre_attention_mlp(augment_dim//2),split_dim=augment_dim//2,event_dim=event_dim)
        wrapped  = PreConditionApplier(temp_flow,temp_coupling_preconditioner_attn)
        transforms.append(wrapped)
    slicer = Slice(distrib_slice,input_dim,dim=-1)
    transforms.append(slicer)
    return Flow(transforms)
        

    
