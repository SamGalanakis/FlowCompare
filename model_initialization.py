import models
import inspect
import torch
import math
import einops
from torch import nn



def set_module_name_tag(module,name_tag):
    """Give name_tag attribute to object and all children"""
    module.name_tag = name_tag
    children = inspect.getmembers(module, lambda x:isinstance(x,torch.nn.Module))
    for class_type,child in children:
        set_module_name_tag(child,name_tag)


def load_flow(load_dict, models_dict):
    """Load flow given initialized mdoel_dict and load_dict checkpoint"""

    models_dict['input_embedder'].load_state_dict(load_dict['input_embedder'])
    models_dict['flow'].load_state_dict(load_dict['flow'])
    return models_dict

def save_flow(model_dict,config,optimizer,scheduler,save_path):
    save_dict = {'config': config._items, "optimizer": optimizer.state_dict(
                    ), "flow": model_dict['flow'].state_dict(), "input_embedder": model_dict['input_embedder'].state_dict(),'scheduler':scheduler.state_dict()}
    torch.save(save_dict, save_path)

def initialize_flow(config, device='cuda', mode='train'):
    """Initialize full model with given config and mode, returns a model_dict"""

    extra_context_dim = 0
    if config['extra_z_value_context']:
        extra_context_dim+=1


    config['extra_context_dim'] = extra_context_dim
    config['using_extra_context'] = True if extra_context_dim>0 else False
    
    # Set global bool as needed for inner loop changes regarding global embedding vs attention
    if config['input_embedder'] in ['DGCNNembedderGlobal']:
        config['global'] = True
    else:
        config['global'] = False


    parameters = []


    #out_dim,query_dim, context_dim, heads, dim_head, dropout
    attn = lambda: models.get_cross_attn(config['attn_dim'], config['attn_input_dim'],
                                             config['input_embedding_dim'], config['cross_heads'], config['cross_dim_head'], config['attn_dropout'])

    if config['coupling_block_nonlinearity'] == "ELU":
        coupling_block_nonlinearity = nn.ELU()
    elif config['coupling_block_nonlinearity'] == "RELU":
        coupling_block_nonlinearity = nn.ReLU()
    elif config['coupling_block_nonlinearity'] == "GELU":
        coupling_block_nonlinearity = nn.GELU()
    else:
        raise Exception("Invalid coupling_block_nonlinearity")

    if config['latent_dim'] > config['input_dim']:

    
        if config['augmenter_dist'] == 'StandardNormal':
            augmenter_dist = models.StandardNormal(
                shape=(config['sample_size'], config['latent_dim']-config['input_dim']))

            augmenter = models.Augment(
            augmenter_dist, split_dim=-1, x_size=config['input_dim'],use_context=False)
        elif config['augmenter_dist'] == 'ConditionalNormal':
            if config['use_attn_augment']: 
                net_augmenter_dist = models.MLP(config['attn_dim']+config['input_dim']+ config['extra_context_dim'], config['net_augmenter_dist_hidden_dims'], (
                    config['latent_dim']-config['input_dim'])*2, coupling_block_nonlinearity)
                augmenter_dist = models.ConditionalNormal(net=net_augmenter_dist,split_dim = -1)
                augmenter_ = models.Augment(
                augmenter_dist, split_dim=-1, x_size=config['input_dim'],use_context=True)
                pre_attn_mlp_ = models.MLP(config['input_dim'],config['hidden_dims'],config['attn_input_dim'],nonlin=coupling_block_nonlinearity)
                augmenter = models.AugmentAttentionPreconditioner(augmenter_,attn,pre_attn_mlp_)
            else:
                net_augmenter_dist = models.MLP(config['input_dim'], config['net_augmenter_dist_hidden_dims'], (
                    config['latent_dim']-config['input_dim'])*2, coupling_block_nonlinearity)
                augmenter_dist = models.ConditionalNormal(net=net_augmenter_dist,split_dim = -1)
                augmenter = models.Augment(
                augmenter_dist, split_dim=-1, x_size=config['input_dim'],use_context=False)
        else:
            raise Exception('Invalid augmenter_dist')

        
    elif config['latent_dim'] == config['input_dim']:
        augmenter = models.IdentityTransform()
    else:
        raise Exception('Latent dim < Input dim')

    if config['flow_type'] == 'AffineCoupling':
        def flow_for_cif(input_dim, context_dim): return models.AffineCoupling(input_dim, context_dim=context_dim,
                                                                               nonlinearity=coupling_block_nonlinearity, hidden_dims=config['hidden_dims'], scale_fn_type=config['affine_scale_fn'])
    elif config['flow_type'] == 'ExponentialCoupling':
        def flow_for_cif(input_dim, context_dim): return models.ExponentialCoupling(input_dim, context_dim=context_dim, nonlinearity=coupling_block_nonlinearity, hidden_dims=config['hidden_dims'],
                                                                                    eps_expm=config['eps_expm'], algo=config['coupling_expm_algo'])
    elif config['flow_type'] == 'RationalQuadraticSplineCoupling':
        def flow_for_cif(input_dim, context_dim): return models.RationalQuadraticSplineCoupling(input_dim, context_dim=context_dim, nonlinearity=coupling_block_nonlinearity, hidden_dims=config['hidden_dims'],
                                                                                                num_bins=config['num_bins_spline']
                                                                                                )

    else:
        raise Exception('Invalid flow type')

    

    def pre_attention_mlp(input_dim_pre_attention_mlp): return models.MLP(input_dim_pre_attention_mlp,
                                                                          config['pre_attention_mlp_hidden_dims'], config['attn_input_dim'], coupling_block_nonlinearity, residual=True)

    if config['permuter_type'] == 'ExponentialCombiner':
        def permuter(dim): return models.ExponentialCombiner(
            dim, eps_expm=config['eps_expm'])
    elif config['permuter_type'] == "random_permute":
        def permuter(dim): return models.Permuter(
            permutation=torch.randperm(dim, dtype=torch.long).to(device))
    elif config['permuter_type'] == "LinearLU":
        def permuter(dim): return models.LinearLU(
            num_features=dim, eps=config['linear_lu_eps'])
    elif config['permuter_type'] == 'FullCombiner':
        def permuter(dim): return models.FullCombiner(dim=dim)
    else:
        raise Exception(
            f'Invalid permuter type: {config["""permuter_type"""]}')


    def cif_block(): return models.cif_helper(config,flow_for_cif, attn,pre_attention_mlp, event_dim=-1)

    transforms = []
    # Add transformations to list

    set_module_name_tag(augmenter,'augmenter')
    transforms.append(augmenter)
    
    
    for index in range(config['n_flow_layers']):
        layer_list = []
        layer_list.append(cif_block())
        # Don't permute output
        if index != config['n_flow_layers']-1:
            if config['act_norm']:
                layer_list.append(models.ActNormBijectionCloud(
                    config['latent_dim'], data_dep_init=True))
            layer_list.append(permuter(config['latent_dim']))
        for module in layer_list:
            set_module_name_tag(module,index)
        transforms.extend(layer_list)
   
    base_dist = models.StandardNormal(
        shape=(config['sample_size'], config['latent_dim']))
    sample_dist = models.Normal(torch.zeros(1), torch.ones(
        1)*0.6, shape=(config['sample_size'], config['latent_dim']))



    final_flow = models.Flow(transforms, base_dist, sample_dist)

    if config['input_embedder'] == 'DGCNNembedder':
        input_embedder = models.DGCNNembedder(
            emb_dim=config['input_embedding_dim'], n_neighbors=config['n_neighbors'], out_mlp_dims=config['hidden_dims_embedder_out'])
    elif config['input_embedder'] == 'PAConv':
        input_embedder = models.PointNet2SSGSeg( c=3,k=config['input_embedding_dim'],out_mlp_dims=config['hidden_dims_embedder_out'])
    elif config['input_embedder'] == 'DGCNNembedderGlobal':
        input_embedder = models.DGCNNembedderGlobal(
            input_dim=config['input_dim'], out_mlp_dims=config['hidden_dims_embedder_out'],
             n_neighbors=config['n_neighbors'], emb_dim=config['input_embedding_dim'])

    elif config['input_embedder'] == 'idenity':
        input_embedder = nn.Identity()
    else:
        raise Exception('Invalid input embeder!')

    if mode == 'train':
        input_embedder.train()
        final_flow.train()

    else:
        input_embedder.eval()
        final_flow.eval()

    if config['data_parallel']:
        input_embedder = nn.DataParallel(input_embedder).to(device)
        final_flow = nn.DataParallel(final_flow).to(device)

    else:
        input_embedder = input_embedder.to(device)
        final_flow = final_flow.to(device)

    parameters += input_embedder.parameters()
    parameters += final_flow.parameters()

    models_dict = {'parameters': parameters,
                   "flow": final_flow, 'input_embedder': input_embedder}

    print(
        f'Number of trainable parameters: {sum([x.numel() for x in parameters])}')
    return models_dict


    
def inner_loop(batch, models_dict, config):

    
    """Computes forward pass of given batch through model, returns mean negative log likelihood loss,log likelihood and bits per dim"""   
    extract_0, extract_1,extra_context = batch

    if extra_context!=None:
        extra_context = einops.repeat(extra_context,'b c-> b n c',n = config['sample_size'])
    
    input_embeddings = models_dict["input_embedder"](extract_0)


    if config['global']:
        input_embeddings = input_embeddings.unsqueeze(1)

    x = extract_1

    log_prob = models_dict['flow'].log_prob(x, context=input_embeddings,extra_context  = extra_context)

    loss = -log_prob.mean()
    with torch. no_grad():
        bpd = loss*math.log2(math.exp(1)) / config['input_dim']
    return loss, log_prob, bpd


def make_sample(n_points, extract_0,models_dict, config, sample_distrib=None,extra_context=None):
    """Computes inverse/generative pass of given model generating n_points given context extract_0,extra_context"""

    input_embeddings = models_dict["input_embedder"](extract_0)

    if extra_context!=None:
        extra_context = einops.repeat(extra_context,'b c-> b n c',n = n_points)

    

    x = models_dict['flow'].sample(num_samples=1, n_points=n_points,
                                   context=input_embeddings, sample_distrib=sample_distrib,extra_context=extra_context).squeeze()
    return x