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

from utils import loss_fun , loss_fun_ret, view_cloud

from data.datasets_pointflow import (
    CIFDatasetDecorator,
    ShapeNet15kPointClouds,
    CIFDatasetDecoratorMultiObject,
)

config_path = "config//config_straight.yaml"
print(f"Loading config from {config_path}")
wandb.init(project="pointflowchange",config = config_path)





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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device {device}')



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


all_params = []
#Get parameters from file and assign them
saved_model_dict = torch.load("save\straight_299_laced-durian-118.pt")
for model_name, model_part in model_dict.items():
    #Send to device before passing to optimizer
    model_part.to(device)
    model_part.load_state_dict(saved_model_dict[model_name])
    #Add model to watch list
    wandb.watch(model_part)
    model_part.eval()
    all_params += model_part.parameters()
# optimizer = Adam(all_params,lr=lr)

# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)


    

#Test
with torch.no_grad():
    x = prior_z.sample(sample_shape = (1,sample_size//5))
    x = x.to(device)
    for g_block in g_blocks[::-1]:
        for g_layer in g_block[::-1]:
            x = g_layer.inverse(x)
    view_cloud(x)
    


    