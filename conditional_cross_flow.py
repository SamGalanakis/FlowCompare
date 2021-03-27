import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,extract_area,co_min_max,feature_assigner,bits_per_dim
from torch.utils.data import Dataset,DataLoader
from torch.distributions.utils import _sum_rightmost
from itertools import permutations, combinations
from tqdm import tqdm
from torch_geometric.data import Data,Batch
from torch_geometric.nn import fps
from dataloaders import ConditionalDataGrid, ShapeNetLoader
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import torch.distributed as distributed
from torch.autograd import Variable, Function
import torch.multiprocessing as mp
from torch_geometric.nn import DataParallel as geomDataParallel
from torch import nn
import functools
import pandas as pd
from time import time,perf_counter
from models import (
Exponential_combiner,
Learned_permuter,
exponential_matrix_coupling_attn,
NeighborhoodEmbedder,
get_cross_attn,
Full_matrix_combiner,
affine_coupling_attn
)


def load_cross_flow(load_dict,initialized_cross_flow):
    initialized_cross_flow['initial_attn']= load_dict['initial_attn']
    initialized_cross_flow['input_embedder'].load_state_dict(load_dict['input_embedder'])
    for layer_dicts,layer in zip(load_dict['layers'],initialized_cross_flow['layers']):
        for key,val in layer.items():
                if isinstance(val,nn.Module):
                    val.load_state_dict(layer_dicts[key])
                elif isinstance(val,pyro.distributions.pyro.distributions.transforms.Permute):
                    val.permutation = layer_dicts[key]
                else:
                    raise Exception('How to load?')
    return initialized_cross_flow



def initialize_cross_flow(config,device = 'cuda',mode='train'):
    flow_input_dim = config['input_dim']
    flow_type = config['flow_type']
    permuter_type = config['permuter_type']
    data_parallel = config['data_parallel']
    parameters = []


    if config['coupling_block_nonlinearity']=="ELU":
        coupling_block_nonlinearity = nn.ELU()
    elif config['coupling_block_nonlinearity']=="RELU":
        coupling_block_nonlinearity = nn.ReLU()
    else:
        raise Exception("Invalid coupling_block_nonlinearity")


    if config['flow_type'] == 'affine_coupling':
        flow = lambda : affine_coupling_attn(config['input_dim'],config['attn_dim'],hidden_dims= config['hidden_dims'])
    elif config['flow_type'] == 'exponential_coupling':
        flow = lambda : exponential_matrix_coupling_attn(config['input_dim'],config['attn_dim'],coupling_block_nonlinearity,hidden_dims= config['hidden_dims'])
    else:
        raise Exception('Invalid flow type')
    #Input size for prev attn + split of point
    attn = lambda : get_cross_attn(config['attn_dim'],config['attn_dim']+config['input_dim']//2,config['input_embedding_dim'],config['cross_heads'],config['cross_dim_head'],config['attn_dropout'])

    if permuter_type == 'Exponential_combiner':
        permuter = lambda : Exponential_combiner(flow_input_dim)
    elif permuter_type == 'Learned_permuter':
        permuter = lambda : Learned_permuter(flow_input_dim)
    elif permuter_type == 'Full_matrix_combiner':
        permuter = lambda : Full_matrix_combiner(flow_input_dim)
    elif permuter_type == "random_permute":
        permuter = lambda : T.Permute(torch.randperm(flow_input_dim, dtype=torch.long).to(device))
    else:
        raise Exception(f'Invalid permuter type: {permuter_type}')

    
    layers = []
    
    #Add transformations to list
    for index in range(config['n_flow_layers']):
        layer_dict = {}
        layer_dict['flow'] = flow()
        layer_dict['attn'] = attn()
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
                if data_parallel:
                    module = nn.DataParallel(module).to(device)
                else:
                    transform = module.to(device)
                parameters+= module.parameters()


    input_embedder = NeighborhoodEmbedder(config['input_dim'])
    if mode == 'train':
        input_embedder.train()
    else:
        input_embedder.eval()
    if data_parallel:
        input_embedder = nn.DataParallel(input_embedder).to(device)
    else:
        input_embedder = input_embedder.to(device)
    
    
    parameters += input_embedder.parameters()
    initial_attn = nn.Parameter(torch.randn((config['attn_dim']),device=device),requires_grad=True)
    parameters += [initial_attn]

    return {'parameters':parameters,"layers":layers,'input_embedder':input_embedder,'initial_attn':initial_attn}
def inner_loop_cross(extract_0,extract_1,models_dict,base_dist,config):

    attn_emb = models_dict['initial_attn'].repeat((extract_0.shape[0],config['sample_size'],1))
    input_embeddings = models_dict["input_embedder"](extract_0)
    y = extract_1
    log_prob = 0.0
    for idx,layer in enumerate(reversed(models_dict['layers'])):
        flow = layer['flow']
        y1, y2 = y.split([flow.split_dim, y.size(flow.dim) - flow.split_dim], dim=flow.dim)
        prev_attn_and_non_change_part = torch.cat((attn_emb,y1),dim=-1)
        attn_emb = layer['attn'](prev_attn_and_non_change_part,context = input_embeddings)
        
        x = flow._inverse(y,attn_emb=attn_emb)
        
        log_prob = log_prob - _sum_rightmost(flow.log_abs_det_jacobian(x, y,attn_emb),
                                        1 - flow.domain.event_dim)
        y=x
        if idx!=(len(models_dict['layers'])-1):
            
            permuter = layer['permuter']
            x = permuter.inv(y)
            log_prob = log_prob - _sum_rightmost(permuter.log_abs_det_jacobian(x, y),
                                        1 - permuter.domain.event_dim)
            y=x
    #Log prob 1 given 0
    log_prob = log_prob + _sum_rightmost(base_dist.log_prob(y),
                                    1 - len(base_dist.event_shape))
    loss = -log_prob.mean()
    return loss,log_prob
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
    else:
        raise Exception('Invalid dataloader type!')

  
    dataloader = DataLoader(dataset,shuffle=True,batch_size=config['batch_size'],num_workers=config["num_workers"],collate_fn=None,pin_memory=True,prefetch_factor=2,drop_last=True)


    base_dist = dist.Normal(torch.zeros(config['input_dim']).to(device), torch.ones(config['input_dim']).to(device))

    models_dict = initialize_cross_flow(config,device,mode='train')
    


    parameters = models_dict['parameters']
  
    
    
    if config["optimizer_type"] =='Adam':
        optimizer = torch.optim.Adam(parameters, lr=config["lr"],weight_decay=config["weight_decay"]) 
    elif config["optimizer_type"] == 'Adamax':
        optimizer = torch.optim.Adamax(parameters, lr=config["lr"],weight_decay=config["weight_decay"],polyak =  0.999)
    elif config["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, lr=config["lr"],weight_decay=config["weight_decay"])
    else:
        raise Exception('Invalid optimizer type!')

    

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=config["patience"],threshold=0.0001,min_lr=config["min_lr"])
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

    def layer_saver(layers):
        dicts = []
        for layer in layers:
            temp_dict = {}
            for key,val in layer.items():
                if isinstance(val,nn.Module):
                    save = val.state_dict()
                elif isinstance(val,pyro.distributions.pyro.distributions.transforms.Permute):
                    save = val.permutation
                else:
                    raise Exception('How to load?')
                temp_dict[key] = save
            dicts.append(temp_dict)
        return dicts
           
    
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    for epoch in range(config["n_epochs"]):
        print(f"Starting epoch: {epoch}")
        for batch_ind,batch in enumerate(tqdm(dataloader)):
            with torch.cuda.amp.autocast(enabled=config['amp']):
                torch.cuda.synchronize()
                t0 = perf_counter()
                batch = [x.to(device) for x in batch]
                extract_0,extract_1 = batch
    
      
                loss, _ = inner_loop_cross(extract_0,extract_1,models_dict,base_dist,config)
    
            
            scaler.scale(loss).backward()
      


            torch.nn.utils.clip_grad_norm_(parameters,max_norm=10)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step(loss)
            optimizer.zero_grad(set_to_none=True)
            current_lr = optimizer.param_groups[0]['lr']
            torch.cuda.synchronize()
            time_batch = perf_counter() - t0

            wandb.log({'loss':loss.item(),'lr':current_lr,'time_batch':time_batch})
          
        
        save_dict = {"optimizer": optimizer.state_dict(),"scheduler":scheduler.state_dict(),"layers":layer_saver(models_dict['layers']),"initial_attn":models_dict['initial_attn'].detach(),"input_embedder":models_dict['input_embedder'].state_dict()}
        torch.save(save_dict,os.path.join(save_model_path,f"{wandb.run.name}_{epoch}_model_dict.pt"))
            
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    rank=''
    main(rank,world_size)