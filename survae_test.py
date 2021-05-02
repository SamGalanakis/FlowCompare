import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.distributions.utils import _sum_rightmost
from tqdm import tqdm
from dataloaders import ConditionalDataGrid, ShapeNetLoader,AmsGridLoader
import wandb
import torch.multiprocessing as mp
from torch import nn
import pandas as pd
import einops
import math
from time import time,perf_counter
from models import (
ExponentialCombiner,
NeighborhoodEmbedder,
get_cross_attn,
ExponentialCoupling,
DGCNNembedder,
GCNembedder,
DGCNNembedderCombo,
MLP,
Permuter,
Augment,
StandardUniform,
StandardNormal,
Slice
)


def load_cross_flow(load_dict,initialized_cross_flow):
    
    initialized_cross_flow['augmenter'].load_state_dict(load_dict['augmenter'])
    initialized_cross_flow['base_dist'].load_state_dict(load_dict['base_dist'])
    initialized_cross_flow['input_embedder'].load_state_dict(load_dict['input_embedder'])

    for layer_dicts,layer in zip(load_dict['layers'],initialized_cross_flow['layers']):
        for key,val in layer.items():
                if isinstance(val,nn.Module):
                    val.load_state_dict(layer_dicts[key])
                else:
                    raise Exception('How to load?')
    return initialized_cross_flow



def initialize_cross_flow(config,device = 'cuda',mode='train'):
    

    parameters = []

    if config['coupling_block_nonlinearity']=="ELU":
        coupling_block_nonlinearity = nn.ELU()
    elif config['coupling_block_nonlinearity']=="RELU":
        coupling_block_nonlinearity = nn.ReLU()
    else:
        raise Exception("Invalid coupling_block_nonlinearity")

    if config['augmenter']:

        if config['augmenter_dist'] == StandardUniform:
            augmenter_dist = StandardUniform(shape = (config['sample_size'],config['latent_dim']-config['input_dim']))
        else: 
            raise Exception('Invalid augmenter_dist')
    

        augmenter_dist = augmenter_dist.to(device)
        parameters += augmenter_dist.parameters()
        augmenter = Augment(augmenter_dist,split_dim=-1,x_size = config['input_dim'])
    else:
        augmenter = torch.nn.Identity().to(device)

    if mode == 'train':
            augmenter.train()
    else:
        augmenter.eval()
    if config['data_parallel']:
        augmenter = nn.DataParallel(augmenter).to(device)
    else:
        augmenter = augmenter.to(device)

    # Is both neccesary??
    parameters+= augmenter.parameters()

    if config['flow_type'] == 'affine_coupling':
        flow = lambda : affine_coupling_attn(config['input_dim'],config['attn_dim'],hidden_dims= config['hidden_dims'])
    elif config['flow_type'] == 'exponential_coupling':
        flow_with_attn = lambda : ExponentialCoupling(input_dim=config['latent_dim'],context_dim = config['attn_dim'],nonlinearity = coupling_block_nonlinearity,hidden_dims= config['hidden_dims'], eps = config['eps_expm'])
        plain_flow = lambda : ExponentialCoupling(input_dim=config['latent_dim'],context_dim = None,nonlinearity = coupling_block_nonlinearity,hidden_dims= config['hidden_dims'], eps = config['eps_expm'])
        
    else:
        raise Exception('Invalid flow type')
    
    
    #out_dim,query_dim, context_dim, heads, dim_head, dropout
    attn = lambda : get_cross_attn(config['attn_dim'],config['attn_input_dim'],config['input_embedding_dim'],config['cross_heads'],config['cross_dim_head'],config['attn_dropout'])

    if config['attn_connection']:
        pre_attention_mlp = lambda : MLP(config['latent_dim']//2 + config['attn_dim'],[config['latent_dim']//2 + config['attn_dim']]*4,config['attn_input_dim'],coupling_block_nonlinearity,residual=True)
        initial_attn_emb = torch.randn(config['attn_dim']).to(device)
        parameters+=initial_attn_emb
    else:
        input_dim_pre_attention_mlp = config['latent_dim']//2
        if config['input_embedder'] == 'DGCNNembedderCombo':
            input_dim_pre_attention_mlp += config['global_input_embedding_dim']
        pre_attention_mlp = lambda : MLP(input_dim_pre_attention_mlp,[config['attn_input_dim']//2]*4,config['attn_input_dim'],coupling_block_nonlinearity,residual=True)
    
    
    if config['permuter_type'] == 'Exponential_combiner':
        permuter = lambda : ExponentialCombiner(config['latent_dim'])
    elif config['permuter_type'] == "random_permute":
        permuter = lambda : Permuter(permutation = torch.randperm(config['latent_dim'], dtype=torch.long).to(device))
    else:
        raise Exception(f'Invalid permuter type: {config["""permuter_type"""]}')

    
    layers = []
    
    #Add transformations to list
    for index in range(config['n_flow_layers']):
        
        layer_dict = {}
        if index % config['attention_frequency'] == 0:
            layer_dict['pre_attention_mlp'] = pre_attention_mlp()
            layer_dict['flow_with_attn'] = flow_with_attn()
            layer_dict['attn'] = attn()
        else:
            layer_dict['plain_flow'] = plain_flow()
        
        #Don't put on first (last on reverse)
        if index != 0:
            layer_dict['permuter'] = permuter()
        layers.append(layer_dict)
        
    for layer in layers:
        for module in layer.values():
            if isinstance(module,torch.nn.Module):
                if mode == 'train':
                    module.train()
                else:
                    module.eval()
                if config['data_parallel']:
                    module = nn.DataParallel(module).to(device)
                else:
                    transform = module.to(device)
                parameters+= module.parameters()


    if config['input_embedder'] == 'NeighborhoodEmbedder':
        input_embedder = NeighborhoodEmbedder(config['input_dim'],out_dim = config['input_embedding_dim'])
    elif config['input_embedder'] == 'DGCNNembedder':
        input_embedder = DGCNNembedder(emb_dim= config['input_embedding_dim'],n_neighbors=config['n_neighbors'])
    elif config['input_embedder'] == 'DGCNNembedderCombo':
        input_embedder = DGCNNembedderCombo(config['input_embedding_dim'],config['global_input_embedding_dim'],n_neighbors=config['n_neighbors'])

    elif config['input_embedder'] == 'idenity':
        input_embedder = nn.Identity()
    else:
        raise Exception('Invalid input embeder!')

    if mode == 'train':
        input_embedder.train()
     
    else:
        input_embedder.eval()
        
    if config['data_parallel']:
        input_embedder = nn.DataParallel(input_embedder).to(device)

    else:
        input_embedder = input_embedder.to(device)

    
  
    parameters += input_embedder.parameters()

    models_dict = {'parameters':parameters,"layers":layers,'input_embedder':input_embedder}
    if config['attn_connection']:
        models_dict['initial_attn_emb'] = initial_attn_emb

    models_dict['augmenter'] = augmenter


    base_dist =  StandardNormal(shape = (config['sample_size'],config['latent_dim']-config['input_dim'])).to(device)
    parameters += base_dist.parameters()
    
    models_dict['base_dist'] = base_dist
    print(f'Number of trainable parameters: {sum([x.numel() for x in parameters])}')
    return models_dict


def inner_loop_cross(extract_0,extract_1,models_dict,config):
    log_prob = 0.0
    if config['attn_connection']:
        attn_emb = models_dict['initial_attn_emb'].repeat(extract_1.shape[0] + (config['latent_dim'],1))

    if not config['input_embedder'] == 'DGCNNembedderCombo':
        input_embeddings = models_dict["input_embedder"](extract_0)
    else: 
        input_embeddings,global_embeddings = models_dict["input_embedder"](extract_0)
        global_embeddings = einops.repeat(global_embeddings,'m n -> m k n', k = extract_1.shape[1])
    x, ldj = models_dict['augmenter'](extract_1,context = None)
    log_prob += ldj
    
    for idx,layer in enumerate(reversed(models_dict['layers'])):
        
        if 'attn' in layer:
            flow=layer['flow_with_attn']
            y1, y2 = y.split([flow.split_dim, y.size(flow.event_dim) - flow.split_dim], dim=flow.event_dim)
            if config['input_embedder'] == 'DGCNNembedderCombo':
                y1 = torch.cat((y1,global_embeddings),dim=-1)
            if config['attn_connection']:
                y1 = torch.cat((y1,attn_emb),dim=-1)
          
            attn_emb = layer['attn'](layer['pre_attention_mlp'](y1),context = input_embeddings)
            
            x,ldj = flow(y,context = attn_emb)
            

            
            log_prob = log_prob - _sum_rightmost(flow.log_abs_det_jacobian(x, y,attn_emb),
                                            1 - flow.domain.event_dim)
        else:
            flow=layer['plain_flow']
            x,ldj = flow(x,context=None)
        
        log_prob += ldj

        if "permuter" in layer:
            
            permuter = layer['permuter']
            x,ldj = permuter(x,context=None)
            log_prob += ldj
    #Log prob 1 given 0
    log_prob += models_dict["base_dist"].log_prob(x)
    loss = -log_prob.sum() / (math.log(2) * x.numel())
    
    return loss,log_prob
def sample_cross(n_samples,extract_0,models_dict,base_dist_for_sample,config):

    input_embeddings = models_dict["input_embedder"](extract_0)

    if not config['input_embedder'] == 'DGCNNembedderCombo':
        input_embeddings = models_dict["input_embedder"](extract_0)
    else: 
        input_embeddings,global_embeddings = models_dict["input_embedder"](extract_0)
        global_embeddings = einops.repeat(global_embeddings,'m n -> m k n', k = n_samples)
    x = base_dist_for_sample.sample(config['batch_size'])
    
    for idx,layer in enumerate(models_dict['layers']):
       
        if "permuter" in layer:
            
            permuter = layer['permuter']
            x = permuter.inverse(x,context=None)
    
        
    
        if 'attn' in layer:
            
            flow=layer['flow_with_attn']
            x1, x2 = x.split([flow.split_dim, x.size(flow.dim) - flow.split_dim], dim=flow.dim)
            if config['input_embedder'] == 'DGCNNembedderCombo':
                x1 = torch.cat((x1,global_embeddings),dim=-1) 
            attn_emb = layer['attn'](layer['pre_attention_mlp'](x1),context = input_embeddings)
            
            x = flow.inverse(x,context=attn_emb)
            
        else:
            flow=layer['plain_flow']
            x = flow.inverse(x,context=None)
        x = models_dict['augmenter'].inverse(x,context=None)
    return x
def layer_saver(layers):
        dicts = []
        for layer in layers:
            temp_dict = {}
            for key,val in layer.items():
                if isinstance(val,nn.Module):
                    save = val.state_dict()
                else:
                    raise Exception('How to load?')
                temp_dict[key] = save
            dicts.append(temp_dict)
        return dicts

def main(rank, world_size):

    dirs = [r'/mnt/cm-nas03/synch/students/sam/data_test/2018',r'/mnt/cm-nas03/synch/students/sam/data_test/2019',r'/mnt/cm-nas03/synch/students/sam/data_test/2020']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    config_path = r"config/config_conditional_cross.yaml"
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config
   

    if config['preselected_points']:
        scene_df_dict = {int(os.path.basename(x).split("_")[0]): pd.read_csv(os.path.join(config['dirs_challenge_csv'],x)) for x in os.listdir(config['dirs_challenge_csv']) }
        preselected_points_dict = {key:val[['x','y']].values for key,val in scene_df_dict.items()}
        preselected_points_dict = { key:(val.unsqueeze(0) if len(val.shape)==1 else val) for key,val in preselected_points_dict.items() }
    else: 
        preselected_points_dict= None

    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path,r"save/processed_dataset")
    if config['data_loader'] == 'ConditionalDataGridSquare':
        dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=config['preload'],subsample=config["subsample"],sample_size=config["sample_size"],min_points=config["min_points"],grid_type='square',normalization=config['normalization'],grid_square_size=config['grid_square_size'],preselected_points=preselected_points_dict)
    elif config['data_loader'] == 'ConditionalDataGridCircle':
        dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'],preselected_points=preselected_points_dict)
    elif config['data_loader']=='ShapeNet':
        dataset = ShapeNetLoader(r'D:\data\ShapeNetCore.v2.PC15k\02691156\train',out_path=out_path,preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'])
    elif config['data_loader'] == 'AmsGridLoader':
        dataset=AmsGridLoader('/media/raid/sam/ams_dataset/',out_path='/media/raid/sam/processed_ams',preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'])

    else:
        raise Exception('Invalid dataloader type!')

  
    dataloader = DataLoader(dataset,shuffle=True,batch_size=config['batch_size'],num_workers=config["num_workers"],collate_fn=None,pin_memory=True,prefetch_factor=2,drop_last=True)


    
    

    models_dict = initialize_cross_flow(config,device,mode='train')
    



    if config["optimizer_type"] =='Adam':
        optimizer = torch.optim.Adam(models_dict['parameters'], lr=config["lr"],weight_decay=config["weight_decay"]) 
    elif config["optimizer_type"] == 'Adamax':
        optimizer = torch.optim.Adamax(models_dict['parameters'], lr=config["lr"],weight_decay=config["weight_decay"],polyak =  0.999)
    elif config["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(models_dict['parameters'], lr=config["lr"],weight_decay=config["weight_decay"])
    else:
        raise Exception('Invalid optimizer type!')

    
    if config['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=config["patience"],threshold=0.0001,min_lr=config["min_lr"])
    elif config['lr_scheduler'] == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = config['lr'], epochs=config['n_epochs'], steps_per_epoch=len(dataloader),total_steps =len(dataloader))
    else:
        raise Exception('Invalid cheduler')
    
    save_model_path = r'save/conditional_flow_compare'

    #Load checkpoint params if specified path
    if config['load_checkpoint']:
        print(f"Loading from checkpoint: {config['load_checkpoint']}")
        checkpoint_dict = torch.load(config['load_checkpoint'])
        models_dict = load_cross_flow(checkpoint_dict,models_dict)
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    else:
        print("Starting training from scratch!")
    #Override min lr to allow for changing after checkpointing
    scheduler.min_lrs = [config['min_lr']]
    #Watch models:


    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])

    for epoch in range(config["n_epochs"]):
        print(f"Starting epoch: {epoch}")
        loss_running_avg = 0
        for batch_ind,batch in enumerate(tqdm(dataloader)):
            with torch.cuda.amp.autocast(enabled=config['amp']):
                torch.cuda.synchronize()
                t0 = perf_counter()
                batch = [x.to(device) for x in batch]
                extract_0,extract_1 = batch
    

                loss, _ = inner_loop_cross(extract_0,extract_1,models_dict,config)
    
            
            scaler.scale(loss).backward()
      


            torch.nn.utils.clip_grad_norm_(models_dict['parameters'],max_norm=config['grad_clip_val'])
            scaler.step(optimizer)
            scaler.update()

            
            optimizer.zero_grad(set_to_none=True)
            current_lr = optimizer.param_groups[0]['lr']
            torch.cuda.synchronize()
            time_batch = perf_counter() - t0
            loss_item = loss.item()
            loss_running_avg = (loss_running_avg*(batch_ind) + loss_item)/(batch_ind+1)
            
            if (batch_ind+1)%config['batches_per_sample'] == 0:
                if not config['attn_connection']:
                    with torch.no_grad():
                        #Multiply std to get tighter samples
                        base_dist_for_sample = torch.distributions.Normal(torch.zeros(config['latent_dim']).to(device), torch.ones(config['latent_dim']).to(device)*0.6)
                        sample = sample_cross(4000,extract_0,models_dict,base_dist_for_sample,config)[0]
                        sample = sample.cpu().numpy().squeeze()
                        sample[:,3:6] = np.clip(sample[:,3:6]*255,0,255)
                        cond_nump = extract_0.cpu().numpy()[0]
                        cond_nump[:,3:6] = np.clip(cond_nump[:,3:6]*255,0,255)
                        wandb.log({"Cond_cloud": wandb.Object3D(cond_nump[:,:6]),"Gen_cloud": wandb.Object3D(sample[:,:6])})
            else:
                wandb.log({'loss':loss_item,'lr':current_lr,'time_batch':time_batch})
            
        scheduler.step(loss_running_avg)
        save_dict = {"optimizer": optimizer.state_dict(),"scheduler":scheduler.state_dict(),"layers":layer_saver(models_dict['layers']),"augmenter":models_dict['augmenter'].state_dict(),"input_embedder":models_dict['input_embedder'].state_dict()}
        if config['attn_connection']:
            save_dict['initial_attn_emb'] = models_dict['initial_attn_emb']
        torch.save(save_dict,os.path.join(save_model_path,f"{wandb.run.name}_{epoch}_model_dict.pt"))
        wandb.log({'epoch':epoch,"loss_epoch":loss_running_avg})
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    rank=''
    main(rank,world_size)