import torch
import torchvision
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
from models.point_encoders import PointnetEncoder, PointnetEncoderNoBatch
from utils import (
    random_subsample,
    loss_fun , 
    loss_fun_ret, 
    view_cloud,
    extract_area,
    load_las
)

from data.datasets_pointflow import (
    CIFDatasetDecorator,
    ShapeNet15kPointClouds,
    CIFDatasetDecoratorMultiObject,
)

config_path = "config//config_straight_with_pointnet.yaml"
print(f"Loading config from {config_path}")
wandb.init(project="pointflowchange",config = config_path)





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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device {device}')


prior_z = distributions.MultivariateNormal(
    torch.zeros(3), torch.eye(3)
)

prior_e = distributions.MultivariateNormal(
    torch.zeros(emb_dim), torch.eye(emb_dim)
)








# Prepare models


pointnet = PointnetEncoderNoBatch(emb_dim,input_dim=3).to(device)




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

all_params = []
for model_part in model_dict.values():
    #Send to device before passing to optimizer
    model_part.to(device)
    #Add model to watch list
    wandb.watch(model_part)
    model_part.train()
    all_params += model_part.parameters()
optimizer = Adam(all_params,lr=lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

points = load_las("D:/data/cycloData/2016/0_5D4KVPBP.las")
sign_point = np.array([86967.46,439138.8])
norm_tranform = torchvision.transforms.Normalize(0,1)
points = extract_area(points,sign_point,1.5,'cylinder')
normtransf = torchvision.transforms.Lambda(lambda x: (x - x.mean(axis=0)) / x.std(axis=0))
norm_stand = torchvision.transforms.Lambda ( lambda x: (x - x.min(axis=0).values) / (x.max(axis=0).values - x.min(axis=0).values)     )
samples = [norm_stand(torch.from_numpy(random_subsample(points,sample_size)[:,:3])) for x in range(4)]
batch = torch.stack(samples).float()
batch = batch.to(device)
for epoch in tqdm(range(n_epochs)):
    loss_acc_z = 0
    loss_acc_e = 0

    optimizer.zero_grad()
   

    #Pass through pointnet
    w = pointnet(batch)
    


    #Pass pointnet embedding through g flow and keep track of determinant
    e_ldetJ = 0
    e = w

    for g_block in g_blocks:
        for g_layer in g_block:
            e, inter_e_ldetJ = g_layer(e)
            e_ldetJ += inter_e_ldetJ
    
    #Pass pointcloud through f flow conditioned on e and keep track of determinant
    z_ldetJ=0
    z = batch
    e = e.expand((batch.shape[1],w.shape[0],w.shape[1])).transpose(0,1)
    for f_block in f_blocks:
        for f_layer in f_block:
            
            
            z, inter_z_ldetJ = f_layer(z,e)
            z_ldetJ += inter_z_ldetJ
    #Undo expanding
    e = e[:,0,:]
    loss_z, loss_e = loss_fun(
            z,
            z_ldetJ,
            prior_z,
            e,
            e_ldetJ,
            prior_e,
        )
    loss = loss_e + loss_z
    loss_acc_z += loss_z.item()
    loss_acc_e += loss_e.item()
    wandb.log({'loss': loss, 'loss_z': loss_z,'loss_e': loss_e})
    loss.backward()
    optimizer.step()
    # Adjust lr according to epoch
    scheduler.step()
    if (epoch % 50 == 0) & (epoch>0):
        
        save_state_dict = {key:val.state_dict() for key,val in model_dict.items()}
        save_state_dict['optimizer'] = optimizer.state_dict()
        save_state_dict['scheduler'] = optimizer.state_dict()
        path_to_save_to = save_model_path+ f"_{epoch}_" +wandb.run.name+".pt"
        print(f"Saving model to {path_to_save_to}")
        torch.save(save_state_dict,path_to_save_to)
        #wandb.save(save_model_path) # File seems to be too big for autosave
    