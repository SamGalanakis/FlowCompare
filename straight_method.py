import torch
import torch.nn as nn
from torch import distributions
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import wandb
import torchvision
from models.flow_modules import (
    CouplingLayer,
    AffineCouplingFunc,
    ConditionalNet,
    StraightNet,
)

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



# cloud_pointflow = ShapeNet15kPointClouds(
#     tr_sample_size=sample_size,
#     te_sample_size=sample_size,
#     root_dir= data_root_dir,
  
#     normalize_per_shape=False,
#     normalize_std_per_axis=False,
#     split="train",
#     scale=1.0,
#     categories=categories,
#     random_subsample=True,
# )



# if random_dataloader:
#         cloud_pointflow = CIFDatasetDecoratorMultiObject(
#             cloud_pointflow, sample_size
#         )
#         batch_size = batch_size 
# dataloader_pointflow = DataLoader(
#     cloud_pointflow, batch_size=batch_size, shuffle=True
# )


# Prepare models







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
for model_part in model_dict.values():
    #Send to device before passing to optimizer
    model_part.to(device)
    #Add model to watch list
    wandb.watch(model_part)
    model_part.train()
    all_params += model_part.parameters()
optimizer = Adam(all_params,lr=lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# base_item = cloud_pointflow[5]
# sample_1 = torch.from_numpy(base_item['points_to_decode'])
# sample_2 = base_item['test_points']
# sample_3 = base_item['train_points']
# batch = torch.stack((sample_1,sample_2,sample_3))   

points = load_las("D:/data/cycloData/2016/0_5D4KVPBP.las")

sign_point = np.array([86967.46,439138.8])
norm_tranform = torchvision.transforms.Normalize(0,1)
points = extract_area(points,sign_point,1.5,'cylinder')
normtransf = torchvision.transforms.Lambda(lambda x: (x - x.mean(axis=0)) / x.std(axis=0))
norm_stand = torchvision.transforms.Lambda ( lambda x: (x - x.min(axis=0).values) / (x.max(axis=0).values - x.min(axis=0).values)     )
samples = [norm_stand(torch.from_numpy(random_subsample(points,1500)[:,:3])) for x in range(4)]
batch = torch.stack(samples).float()
for epoch in tqdm(range(n_epochs)):
    loss_acc_z = 0
    

    optimizer.zero_grad()
    #Add noise to tr_batch:
    batch = batch.to(device) + x_noise * torch.rand(batch.shape).to(device)



    x = batch
    #Pass pointcloud through g flow conditioned and keep track of determinant
    ldetJ=0
    for g_block in g_blocks:
        for g_layer in g_block:
            x, inter_ldetJ = g_layer(x)
            ldetJ += inter_ldetJ
    z = x
    loss = loss_fun_ret(z,ldetJ,prior_z)
    
    wandb.log({'loss':loss.item()})
    loss.backward()
    optimizer.step()
    scheduler.step()
    # Adjust lr according to epoch
    
   


#Save model
save_state_dict = {key:val.state_dict() for key,val in model_dict.items()}
save_state_dict['optimizer'] = optimizer.state_dict()
save_state_dict['scheduler'] = optimizer.state_dict()
save_model_path = save_model_path+ f"_{epoch}_" +wandb.run.name+".pt"
print(f"Saving model to {save_model_path}")
torch.save(save_state_dict,save_model_path)

#Test
with torch.no_grad():
    x = prior_z.sample(sample_shape = (1,sample_size*10))
    x = x.to(device)
    for g_block in g_blocks[::-1]:
        for g_layer in g_block[::-1]:
            x = g_layer.inverse(x)
    view_cloud(x)
    

    