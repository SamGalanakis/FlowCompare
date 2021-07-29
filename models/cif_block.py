import torch
import torch.nn as nn
import models


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
        attn_emb = torch.utils.checkpoint.checkpoint(
            self.attn, mlp_out, context, preserve_rng_state=False)
        return attn_emb


class CouplingPreconditionerGlobal(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, context):
        return context


def cif_helper(config,flow, attn,pre_attention_mlp, event_dim=-1):
    # CIF if aug>base latent dim else normal flow
    if config['latent_dim'] < config['cif_latent_dim']:
        if config['using_extra_context']:
            raise Exception('Not implemented extra context with cif')
        if config['global']:
            raise Exception('CIF + global embedding not implemented')
        else:
            return CIFblock(config,flow,attn,event_dim)
    elif config['latent_dim'] == config['cif_latent_dim']:
        if not config['global']:
            return models.PreConditionApplier(flow(config['latent_dim'], config['attn_dim']+ config['extra_context_dim']), CouplingPreconditionerAttn(attn(), pre_attention_mlp(config['latent_dim']//2), config['latent_dim']//2, event_dim=event_dim))
        else:
            return models.PreConditionApplier(flow(config['latent_dim'], config['input_embedding_dim']+config['extra_context_dim']),CouplingPreconditionerGlobal())

    else:
        raise Exception('Augment dim smaller than main latent!')


class CIFblock(models.Transform):
    def __init__(self, config,flow,attn,event_dim):
        super().__init__()
        self.config = config
        self.event_dim = event_dim
    
        distrib_augment_net = models.MLP(config['latent_dim'],config['net_cif_dist_hidden_dims'],(config['cif_latent_dim']- config['latent_dim'])*2,nonlin=torch.nn.GELU())
        distrib_augment = models.ConditionalNormal(net =distrib_augment_net,split_dim = event_dim,clamp = config['clamp_dist'])
        self.act_norm = models.ActNormBijectionCloud(config['cif_latent_dim'])
        distrib_slice = distrib_augment
        self.augmenter = models.Augment(
            distrib_augment, config['latent_dim'], split_dim=event_dim)
        
        pre_attention_mlp = models.MLP(config['latent_dim']//2,config['pre_attention_mlp_hidden_dims'], config['attn_input_dim'], torch.nn.GELU(), residual=True)

        self.affine_cif = models.AffineCoupling(config['cif_latent_dim'],config['affine_cif_hidden'],nn.GELU(),scale_fn_type='sigmoid',split_dim=config['cif_latent_dim']-config['latent_dim'])
        self.flow = models.PreConditionApplier(flow(config['latent_dim'], config['attn_dim']), CouplingPreconditionerAttn(attn(), pre_attention_mlp, config['latent_dim']//2, event_dim=event_dim))
        self.slicer = models.Slice(distrib_slice, config['latent_dim'], dim=self.event_dim)
        
        self.reverse = models.Reverse(config['cif_latent_dim'],dim=-1)
        

    def forward(self, x, context=None,extra_context=None):
        ldj_cif = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)

        x, ldj = self.augmenter(x, context=None)
        ldj_cif += ldj

        x,_ = self.reverse(x)

        x,ldj = self.affine_cif(x,context=None)
        ldj_cif += ldj

        x,ldj = self.act_norm(x)
        ldj_cif += ldj

        x,_ = self.reverse(x)
        
        x, ldj = self.slicer(x, context=None)
        ldj_cif += ldj


        x, ldj = self.flow(x, context=context)
        ldj_cif += ldj
        
        


        
        

        return x, ldj_cif

    def inverse(self, y, context=None,extra_context=None):
        y = self.flow.inverse(y,context=context)
        y = self.slicer.inverse(y)
        y = self.reverse.inverse(y)
        y = self.act_norm.inverse(y)
        y = self.affine_cif.inverse(y)
        y = self.reverse.inverse(y)
        x = self.augmenter.inverse(y)


        return x
