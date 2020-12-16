import torch
import torch.nn as nn
from torch import distributions
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import wandb
from models.flow_modules import (
    CouplingLayer,
    AffineCouplingFunc,
    ConditionalNet,
    StraightNet,
)
from models.point_encoders import PointnetEncoder

from utils import loss_fun , loss_fun_ret, view_cloud




def model_init(config_path,model_path,model_type,test=True):
    
    #Load config
    print(f"Loading config from {config_path}")
    wandb.init(config = config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')


    if model_type == "straight":
        n_g= wandb.config['n_g'] 
        n_g_k = wandb.config['n_g_k']
        data_root_dir = wandb.config['data_root_dir']
        save_model_path = wandb.config['save_model_path']
        n_epochs = wandb.config['n_epochs']
        sample_size = wandb.config['sample_size']
        batch_size = wandb.config['batch_size']
        x_noise = wandb.config['x_noise']
        random_dataloader = wandb.config['random_dataloader']

        categories = wandb.config['categories']
        lr= wandb.config['lr']


        prior_z = distributions.MultivariateNormal(
            torch.zeros(3), torch.eye(3)
        )


        # for g
        g_blocks = [[] for x in range(n_g)]

        g_permute_list_list = [[2,0,1]]*n_g
        g_split_index_list = [2]*len(g_permute_list_list)

        for i in range(n_g):
            for j in range(n_g_k):
                split_index = g_split_index_list[j]
                permute_tensor = torch.LongTensor([g_permute_list_list[j] ]).to(device)        
                mutiply_func = StraightNet(in_dim = split_index,out_dim=3-split_index)
                add_func  = StraightNet(in_dim = split_index,out_dim=3-split_index)
                coupling_func = AffineCouplingFunc(mutiply_func,add_func)
                coupling_layer = CouplingLayer(coupling_func,split_index,permute_tensor)
                g_blocks[i].append(coupling_layer) 

            
        

        model_dict = {}
        for i, g_block in enumerate(g_blocks):
            for k in range(len(g_block)):
                model_dict[f'g_block_{i}_{k}'] = g_block[k]

        

        

        # model_dict['scheduler'] = scheduler
        # model_dict['optimizer'] = optimizer



    elif model_type == 'straight_with_pointnet':
        n_f= wandb.config['n_f'] 
        n_f_k = wandb.config['n_f_k']
        data_root_dir = wandb.config['data_root_dir']
        save_model_path = wandb.config['save_model_path']
        n_g = wandb.config['n_g']
        n_g_k= wandb.config['n_g_k']
        n_epochs = wandb.config['n_epochs']
        sample_size = wandb.config['sample_size']
        batch_size = wandb.config['batch_size']
        x_noise = wandb.config['x_noise']
        random_dataloader = wandb.config['random_dataloader']
        emb_dim =  wandb.config['emb_dim']
        categories = wandb.config['categories']
        lr= wandb.config['lr']


        prior_z = distributions.MultivariateNormal(
        torch.zeros(3), torch.eye(3)
        )

        prior_e = distributions.MultivariateNormal(
        torch.zeros(emb_dim), torch.eye(emb_dim)
        )

        pointnet = PointnetEncoder(emb_dim,input_dim=3).to(device)

    
        # for f
        f_blocks = [[] for x in range(n_f)]
        f_permute_list_list = [[2,0,1]]*n_f_k
        f_split_index_list = [2]*len(f_permute_list_list)

        for i in range(n_f):
            for j in range(n_f_k):
                split_index = f_split_index_list[j]
                permute_tensor = torch.LongTensor([f_permute_list_list[j] ]).to(device)
                mutiply_func = ConditionalNet(emb_dim=emb_dim,in_dim=split_index,out_dim=3-split_index)
                add_func  = ConditionalNet(emb_dim=emb_dim,in_dim=split_index,out_dim=3-split_index)
                coupling_func = AffineCouplingFunc(mutiply_func,add_func)
                coupling_layer = CouplingLayer(coupling_func,split_index,permute_tensor)
                f_blocks[i].append(coupling_layer)

        # for g
        g_blocks = [[] for x in range(n_g)]
        g_permute_list_list = [list(range(emb_dim//2,emb_dim))+list(range(emb_dim//2))]*n_g_k
        g_split_index_list = [emb_dim//2]*len(g_permute_list_list)

        for i in range(n_g):
            for j in range(n_g_k):
                split_index = g_split_index_list[j]
                permute_tensor = torch.LongTensor([g_permute_list_list[j] ]).to(device)        
                mutiply_func = StraightNet(in_dim = emb_dim//2,out_dim=emb_dim//2)
                add_func  = StraightNet(in_dim = emb_dim//2,out_dim=emb_dim//2)
                coupling_func = AffineCouplingFunc(mutiply_func,add_func)
                coupling_layer = CouplingLayer(coupling_func,split_index,permute_tensor)
                g_blocks[i].append(coupling_layer)        
        

        model_dict = {'pointnet':pointnet}
        for i, f_block in enumerate(f_blocks):
            for k in range(len(f_block)):
                model_dict[f'f_block_{i}_{k}'] = f_block[k]

        for i, g_block in enumerate(g_blocks):
            for k in range(len(g_block)):
                model_dict[f'g_block_{i}_{k}'] = g_block[k]






    #Load model parts from saved dict in same way for all model_types
    all_params = []
    #Get parameters from file and assign them
    saved_model_dict = torch.load(model_path)
    for model_name, model_part in model_dict.items():
        #Send to device before passing to optimizer
        model_part.to(device)
        model_part.load_state_dict(saved_model_dict[model_name])
        #Add model to watch list
        
        if test:
            model_part.eval()
        else:
            wandb.watch(model_part)
            all_params += model_part.parameters()
            



            

    return model_dict
        
            

        