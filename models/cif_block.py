from models import Augment,Slice, PreConditionApplier, IdentityTransform, ActNormBijectionCloud
import torch
import torch.nn as nn 











class CouplingPreconditionerAttn(nn.Module):
    def __init__(self,attn,pre_attention_mlp,noise_dim,x1_dim,event_dim=-1):
        super().__init__()
        self.attn = attn
        self.pre_attention_mlp = pre_attention_mlp
        self.event_dim = event_dim
        self.noise_dim = noise_dim
        self.x1_dim = x1_dim
    def forward(self,x,context):
        x1,x2,noise = x.split([self.x1_dim, self.x1_dim,self.noise_dim], dim=self.event_dim)

        attn_emb = self.attn(self.pre_attention_mlp(torch.cat((x1,noise),dim=self.event_dim)),context = context)
        return attn_emb

def get_cif_block_attn(input_dim,augment_dim,distribution,context_dim,flow,attn,pre_attention_mlp,n_flows,event_dim,permuter,act_norm):
    transforms = []
    
    if input_dim<augment_dim:
        distrib_augment = distribution()
        distrib_slice = distribution()
        augmenter =  Augment(distrib_augment,input_dim,split_dim=event_dim)
        transforms.append(augmenter)
    elif input_dim>augment_dim:
        raise Exception('Input dim larger than augment dim!')
    
    pre_attention_mlp_input_dim = input_dim//2  + augment_dim  #Context is x1,noise

    for index,x in enumerate(range(n_flows)): 
        temp_flow = flow(input_dim=input_dim,context_dim = context_dim) 
        temp_coupling_preconditioner_attn = CouplingPreconditionerAttn(attn(),pre_attention_mlp(pre_attention_mlp_input_dim),noise_dim=augment_dim,x1_dim=input_dim//2,event_dim=event_dim)
        wrapped  = PreConditionApplier(temp_flow,temp_coupling_preconditioner_attn)
        transforms.append(wrapped)
        if index != n_flows-1:
            if act_norm:
                transforms.append(ActNormBijectionCloud(input_dim,data_dep_init=True))
            transforms.append(permuter(input_dim))
    if input_dim<augment_dim:
        slicer = Slice(distrib_slice,input_dim,dim=-1)
        transforms.append(slicer)
    return transforms
        

    
