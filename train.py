import torch
import torch.nn as nn
from torch import distributions
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.flow_modules import (
    CouplingLayer,
    AffineCouplingFunc,
    ConditionalNet,
    StraightNet,
)
from models.point_encoders import PointnetEncoder
from utils import loss_fun , loss_fun_ret, view_cloud

from data.datasets_pointflow import (
    CIFDatasetDecorator,
    ShapeNet15kPointClouds,
    CIFDatasetDecoratorMultiObject,
)
n_f= 10
n_f_k = 3
n_g = 3
n_g_k=2
n_epochs = 100
batch_size =2
random_dataloader = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prior_z = distributions.MultivariateNormal(
    torch.zeros(3), torch.eye(3)
)
emb_dim = 32
prior_e = distributions.MultivariateNormal(
    torch.zeros(emb_dim), torch.eye(emb_dim)
)

cloud_pointflow = ShapeNet15kPointClouds(
    tr_sample_size=2048,
    te_sample_size=2048,
    root_dir='data\\ShapeNetCore.v2.PC15k',
  
    normalize_per_shape=False,
    normalize_std_per_axis=False,
    split="train",
    scale=1.0,
    categories=["airplane"],
    random_subsample=True,
)



if random_dataloader:
        cloud_pointflow = CIFDatasetDecoratorMultiObject(
            cloud_pointflow, 2048
        )
        batch_size = 50
dataloader_pointflow = DataLoader(
    cloud_pointflow, batch_size=batch_size, shuffle=True
)


# Prepare models


pointnet = PointnetEncoder(emb_dim,input_dim=3).to(device)




# for f
f_blocks = [[] for x in range(n_f)]
f_permute_list_list = [[0,2,1],[2,0,1],[1,0,2]]
f_split_index_list = [1]*len(f_permute_list_list)

for i in range(n_f):
    for j in range(n_f_k):
        split_index = f_split_index_list[j]
        permute_list = f_permute_list_list[j]

        mutiply_func = ConditionalNet(emb_dim=emb_dim,in_dim=split_index+1)
        add_func  = ConditionalNet(emb_dim=emb_dim,in_dim = split_index+1)
        coupling_func = AffineCouplingFunc(mutiply_func,add_func)
        coupling_layer = CouplingLayer(coupling_func,split_index,permute_list)
        f_blocks[i].append(coupling_layer)

# for g
g_blocks = [[] for x in range(n_g)]
g_permute_list_list = [list(range(emb_dim))[::-1],[x if x%2 ==0 else emb_dim - x for x in range(emb_dim) ]]
g_split_index_list = [emb_dim//2]*len(g_permute_list_list)

for i in range(n_g):
    for j in range(n_g_k):
        split_index = g_split_index_list[j]
        permute_list = g_permute_list_list[j]        
        mutiply_func = StraightNet(in_dim = split_index+1)
        add_func  = StraightNet(split_index+1)
        coupling_func = AffineCouplingFunc(mutiply_func,add_func)
        coupling_layer = CouplingLayer(coupling_func,split_index,permute_list)
        g_blocks[i].append(coupling_layer)        
   

model_dict = {'pointnet':pointnet}
for i, f_block in enumerate(f_blocks):
    for k in range(len(f_block)):
        model_dict[f'f_block_{i}_{k}'] = f_block

for i, g_block in enumerate(g_blocks):
     for k in range(len(g_block)):
        model_dict[f'g_block_{i}_{k}'] = g_block


for i in range(n_epochs):
    pass
