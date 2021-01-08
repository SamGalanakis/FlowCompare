
import numpy as np
from models.model_init import model_init
import torch
import wandb
from tqdm import tqdm
from torch.optim import Adam, SGD
from utils import (

loss_fun_ret
)


config_path = "config/config_straight.yaml"
model_type = 'straight'
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
device = 'cpu'
prior_z = torch.distributions.MultivariateNormal(
    torch.zeros(3), torch.eye(3)
)


model_dict = model_init(config_path,test=True)

all_params = []
for model_part in model_dict.values():
    #Send to device before passing to optimizer
    model_part.to(device)
    #Add model to watch list
    wandb.watch(model_part)
    model_part.train()
    all_params += model_part.parameters()
optimizer = Adam(all_params,lr=lr)
#optimizer = SGD(all_params,lr=lr)




g_layers = [model_dict[key] for key in sorted(model_dict.keys()) if 'g_block' in key]
f_layers = [model_dict[key] for key in sorted(model_dict.keys()) if 'f_block' in key]




dataset = torch.randn((100000,3)).to(device)

#Warm up cuda

batch = dataset[0*batch_size:(0+1)*batch_size,:].unsqueeze(0)
with torch.no_grad():
    for g_layer in g_layers:
                batch, inter_ldetJ = g_layer(batch)



batch_size=100
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
  

for epoch in tqdm(range(n_epochs)):
    ind = epoch % dataset.shape[0]
    batch = dataset[ind*batch_size:(ind+1)*batch_size,:].unsqueeze(0)
    
    
    loss_acc_z = 0
    

    optimizer.zero_grad()
    #Add noise to tr_batch:
    batch = batch.to(device) #+ x_noise * torch.rand(batch.shape).to(device)



    x = batch
    #Pass pointcloud through g flow conditioned and keep track of determinant
    ldetJ=0

    for g_layer in g_layers:
        x, inter_ldetJ = g_layer(x)
        ldetJ += inter_ldetJ
    z = x
    loss = loss_fun_ret(z,ldetJ,prior_z)
    



    wandb.log({'loss':loss.item()})
    loss.backward()
    optimizer.step()



