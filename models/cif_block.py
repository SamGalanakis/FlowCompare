import torch
import torch.nn as nn

from models import (ActNormBijectionCloud, Augment, IdentityTransform,
                    PreConditionApplier, Reverse, Slice, transform)
from models.transform import Transform
import models

class AffineCif(Transform):
    def __init__(self,input_dim,parameter_mlp_in_dim,hidden_mlp_dims):
        self.input_dim = input_dim
        self.parameter_mlp_in_dim = parameter_mlp_in_dim
        self.hidden_mlp_dims = hidden_mlp_dims

        self.param_mlp = models.MLP(parameter_mlp_in_dim,hidden_mlp_dims,input_dim*2,nonlune=nn.GELU())
        
    def forward(self,x,u):
        s,t = torch.chunk(self.param_mlp(u),2,dim=-1)
        s = torch.exp(-s)
        ldj = torch.einsum('bnd->bn', torch.log(s))
        x = s*x - t
        return x,ldj

    def inverse(self,y,u):
        s,t = torch.chunk(self.param_mlp(u),2,dim=-1)
        s = torch.exp(-s)
        return (y+t)/s



class CouplingPreconditionerNoAttn(nn.Module):
    def __init__(self, event_dim=-1):
        super().__init__()

    def forward(self, x, context):

        return context


class CouplingPreconditionerAttn(nn.Module):
    def __init__(self, attn, pre_attention_mlp, x1_dim, event_dim=-1):
        super().__init__()
        self.attn = attn
        self.pre_attention_mlp = pre_attention_mlp
        self.event_dim = event_dim
        self.x1_dim = x1_dim

    def forward(self, x, context):
        x1, x2 = x.split([self.x1_dim, self.x1_dim], dim=self.event_dim)
        mlp_out = torch.utils.checkpoint.checkpoint(
            self.pre_attention_mlp, x1, preserve_rng_state=False)
        #attn_emb = self.attn(mlp_out,context)
        attn_emb = torch.utils.checkpoint.checkpoint(
            self.attn, mlp_out, context, preserve_rng_state=False)
        return attn_emb


def cif_helper(input_dim, augment_dim, distribution_aug, distribution_slice, context_dim, flow, attn, pre_attention_mlp, event_dim, conditional_aug, conditional_slice, input_embedder_type):
    # CIF if aug>base latent dim else normal flow
    if input_dim < augment_dim:

        if input_embedder_type != 'DGCNNembedderGlobal':
            raise Exception('CIF + global embedding not implemented')

        return CIFblock(input_dim, augment_dim, distribution_aug, distribution_slice, context_dim, flow, attn, pre_attention_mlp, event_dim=event_dim, conditional_aug=conditional_aug, conditional_slice=conditional_slice)
    elif input_dim == augment_dim:
        if input_embedder_type != 'DGCNNembedderGlobal':
            return PreConditionApplier(flow(input_dim, context_dim), CouplingPreconditionerAttn(attn(), pre_attention_mlp(input_dim//2), input_dim//2, event_dim=event_dim))
        else:
            return flow(input_dim, context_dim)

    else:
        raise Exception('Augment dim smaller than main latent!')


class CIFblock(Transform):
    def __init__(self, input_dim, augment_dim, distribution_aug, distribution_slice, context_dim, flow, attn, pre_attention_mlp, event_dim, conditional_aug=True, conditional_slice=True, share_attn_weights=True):
        super().__init__()
        self.input_dim = input_dim
        self.augment_dim = augment_dim
        self.event_dim = event_dim
        self.conditional_aug = conditional_aug
        self.conditional_slice = conditional_slice

        distrib_augment = distribution_aug()
        distrib_slice = distribution_slice()
        self.augmenter = Augment(
            distrib_augment, input_dim, split_dim=event_dim)
        self.affine_cif = AffineCif()
        pre_attention_mlp_input_dim = augment_dim - input_dim  # - input_dim//2  #Context is x1,noise
        self.attention = attn()
        self.pre_attention_mlp = pre_attention_mlp(pre_attention_mlp_input_dim)
        self.flow = flow(input_dim=augment_dim, context_dim=context_dim)
        self.slicer = Slice(distrib_slice, input_dim, dim=self.event_dim)
        self.reverse = Reverse(augment_dim, dim=self.event_dim)

        if self.conditional_aug:
            self.pre_attention_mlp_aug = pre_attention_mlp(input_dim)
            self.attention_aug = attn()

        if self.conditional_slice:
            self.pre_attention_mlp_slice = pre_attention_mlp(input_dim)
            self.attention_slice = attn()

    def forward(self, x, context=None):
        ldj_cif = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)

        if self.conditional_aug:
            aug_mlp_out = torch.utils.checkpoint.checkpoint(
                self.pre_attention_mlp_aug, x, preserve_rng_state=False)
            context_aug = torch.utils.checkpoint.checkpoint(
                self.attention_aug, aug_mlp_out, context, preserve_rng_state=False)
        else:
            context_aug = None

        combined, ldj = self.augmenter(x, context=context_aug)
        ldj_cif += ldj
        x, noise = combined.split(
            [self.input_dim, self.augment_dim-self.input_dim], dim=self.event_dim)

        mlp_out = torch.utils.checkpoint.checkpoint(
            self.pre_attention_mlp, noise, preserve_rng_state=False)
        attention_emb = torch.utils.checkpoint.checkpoint(
            self.attention, mlp_out, context, preserve_rng_state=False)

        x, _ = self.reverse(combined, context=None)  # 0 ldj
        x, ldj = self.flow(x, context=attention_emb)
        ldj_cif += ldj

        x, _ = self.reverse(x, context=None)

        if self.conditional_slice:
            slice_mlp_out = torch.utils.checkpoint.checkpoint(
                self.pre_attention_mlp_slice, x[..., :self.input_dim], preserve_rng_state=False)
            context_slice = torch.utils.checkpoint.checkpoint(
                self.attention_slice, slice_mlp_out, context, preserve_rng_state=False)
        else:
            context_slice = None

        x, ldj = self.slicer(x, context=context_slice)
        ldj_cif += ldj

        return x, ldj_cif

    def inverse(self, y, context=None):

        context_slice = self.attention_slice(self.pre_attention_mlp_slice(
            y), context=context) if self.conditional_slice else None
        y = self.slicer.inverse(y, context=context_slice)
        y = self.reverse.inverse(y, context=None)

        noise, _ = torch.split(
            y, [self.augment_dim-self.input_dim, self.input_dim], dim=self.event_dim)
        attention_emb = self.attention(
            self.pre_attention_mlp(noise), context=context)

        y = self.flow.inverse(y, context=attention_emb)
        y = self.reverse.inverse(y, context=None)

        y = self.augmenter.inverse(y, context=None)

        return y
