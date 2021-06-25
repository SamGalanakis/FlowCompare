import math
import os
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
import wandb
from dataloaders import AmsVoxelLoader
from utils import Scheduler, is_valid


def load_flow(load_dict, models_dict):

    models_dict['input_embedder'].load_state_dict(load_dict['input_embedder'])
    models_dict['flow'].load_state_dict(load_dict['flow'])
    return models_dict


def initialize_flow(config, device='cuda', mode='train'):

    
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
                net_augmenter_dist = models.MLP(config['attn_dim']+config['input_dim'], config['net_augmenter_dist_hidden_dims'], (
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
    transforms.append(augmenter)

    # transforms.append(ActNormBijectionCloud(config['latent_dim'],data_dep_init=True)) #Entry norm
    for index in range(config['n_flow_layers']):

        transforms.append(cif_block())
        # Don't permute output
        if index != config['n_flow_layers']-1:
            if config['act_norm']:
                transforms.append(models.ActNormBijectionCloud(
                    config['latent_dim'], data_dep_init=True))
            transforms.append(permuter(config['latent_dim']))
   
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


    
def inner_loop(extract_0, extract_1, models_dict, config):

    input_embeddings = models_dict["input_embedder"](extract_0)


    if config['global']:
        input_embeddings = input_embeddings.unsqueeze(1)

    x = extract_1

    log_prob = models_dict['flow'].log_prob(x, context=input_embeddings)

    loss = -log_prob.mean()
    with torch. no_grad():
        bpd = loss*math.log2(math.exp(1)) / config['input_dim']
    return loss, log_prob, bpd


def make_sample(n_points, extract_0, models_dict, config, sample_distrib=None):

    input_embeddings = models_dict["input_embedder"](extract_0[0].unsqueeze(0))


    if config['global']:
        input_embeddings = input_embeddings.unsqueeze(1)

    x = models_dict['flow'].sample(num_samples=1, n_points=n_points,
                                   context=input_embeddings, sample_distrib=sample_distrib).squeeze()
    return x


def main():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = r"config/config.yaml"
    wandb.init(project="flow_change", config=config_path)
    config = wandb.config


    models_dict = initialize_flow(config, device, mode='train')


    
    
    if config['data_loader'] == 'AmsVoxelLoader':
        dataset = AmsVoxelLoader(config['directory_path_train'],config['directory_path_test'], out_path='save/processed_dataset', preload=config['preload'],
        n_samples = config['sample_size'],n_voxels=config['batch_size'],final_voxel_size = config['final_voxel_size'],device=device,
        n_samples_context = config['n_samples_context'], context_voxel_size = config['context_voxel_size'],mode='train'
        )
     
    else:
        raise Exception('Invalid dataloader type!')


    batch_size = config['batch_size']
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=config[
                            "num_workers"], collate_fn=None, pin_memory=True, prefetch_factor=2, drop_last=True)

    if config["optimizer_type"] == 'Adam':
        optimizer = torch.optim.Adam(
            models_dict['parameters'], lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == 'Adamax':
        optimizer = torch.optim.Adamax(
            models_dict['parameters'], lr=config["lr"], weight_decay=config["weight_decay"], polyak=0.999)
    elif config["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(
            models_dict['parameters'], lr=config["lr"], weight_decay=config["weight_decay"])
    elif config['optimizer_type'] == 'SGD':
        optimizer = torch.optim.SGD(models_dict['parameters'], lr=config["lr"],
                                    momentum=0, dampening=0, weight_decay=config["weight_decay"], nesterov=False)
    else:
        raise Exception('Invalid optimizer type!')


    scheduler = Scheduler(optimizer, config['mem_iter_scheduler'], factor=config['lr_factor'],
                              threshold=config['threshold_scheduler'], min_lr=config["min_lr"])


    save_model_path = r'save/conditional_flow_compare'

    # Load checkpoint params if specified path
    if config['load_checkpoint']:
        print(f"Loading from checkpoint: {config['load_checkpoint']}")
        checkpoint_dict = torch.load(config['load_checkpoint'])
        models_dict = load_flow(checkpoint_dict, models_dict)

    else:
        print("Starting training from scratch!")

    # Watch models:
    detect_anomaly = False
    if detect_anomaly:
        print('DETECT ANOMALY ON')
        torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    best_so_far = math.inf
    last_save_path = None
    for epoch in range(config["n_epochs"]):
        print(f"Starting epoch: {epoch}")
        loss_running_avg = 0
        for batch_ind, batch in enumerate(tqdm(dataloader)):
            with torch.cuda.amp.autocast(enabled=config['amp']):
                if config['time_stats']:
                    torch.cuda.synchronize()
                    t0 = perf_counter()
                batch = [x.to(device) for x in batch]
          
                extract_0, extract_1 = batch
                loss, _, nats = inner_loop(
                    extract_0, extract_1, models_dict, config)
                is_valid(loss)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                models_dict['parameters'], max_norm=config['grad_clip_val'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(loss)

            optimizer.zero_grad(set_to_none=True)
            current_lr = optimizer.param_groups[0]['lr']
            if config['time_stats']:
                torch.cuda.synchronize()
                time_batch = perf_counter() - t0
            else:
                time_batch = np.NaN
            loss_item = loss.item()
            loss_running_avg = (
                loss_running_avg*(batch_ind) + loss_item)/(batch_ind+1)

            if ((batch_ind+1) % config['batches_per_sample'] == 0) and config['make_samples']:
                with torch.no_grad():
                    
                    cond_nump = extract_0[0].cpu().numpy()
                    sample_points = make_sample(
                        4000, extract_0, models_dict, config)
                    cond_nump[:, 3:6] = np.clip(
                    cond_nump[:, 3:6]*255, 0, 255)
                    sample_points = sample_points.cpu().numpy().squeeze()
                    sample_points[:, 3:6] = np.clip(
                    sample_points[:, 3:6]*255, 0, 255)
                    wandb.log({"Cond_cloud": wandb.Object3D(cond_nump[:, :6]), "Gen_cloud": wandb.Object3D(
                        sample_points[:, :6]), 'loss': loss_item, 'nats': nats.item(), 'lr': current_lr, 'time_batch': time_batch})
            else:
                pass
                wandb.log({'loss': loss_item, 'nats': nats.item(),
                          'lr': current_lr, 'time_batch': time_batch})
            
                
        wandb.log({'epoch': epoch, "loss_epoch": loss_running_avg})
        print(f'Loss epoch: {loss_running_avg}')
        
        
        if (epoch % config['epochs_per_save'])==0 and epoch>0:
            if loss_running_avg < best_so_far:
                if last_save_path!=None:
                    os.remove(last_save_path)
                print(f'Saving!')
                savepath = os.path.join(
                    save_model_path, f"{wandb.run.name}_e{epoch}_model_dict.pt")
                
                save_dict = {'config': config._items, "optimizer": optimizer.state_dict(
                ), "flow": models_dict['flow'].state_dict(), "input_embedder": models_dict['input_embedder'].state_dict()}
                torch.save(save_dict, savepath)
                last_save_path = savepath
        
                best_so_far = min(loss_running_avg,best_so_far)


if __name__ == "__main__":
    main()
