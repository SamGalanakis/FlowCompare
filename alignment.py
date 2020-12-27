from sklearn.datasets import make_swiss_roll
import numpy as np
from models.model_init import model_init
from models.rotation_model import Rotator
import torch
import wandb
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.nn import KLDivLoss
from utils import (
load_las,
compare_clouds,
extract_area,
random_subsample,
compare_clouds,
view_cloud_plotly,
loss_fun_ret,
rotate_mat
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

prior_z = torch.distributions.MultivariateNormal(
    torch.zeros(3), torch.eye(3)
)

model_dict = model_init(config_path,test=True)


for model_part in model_dict.values():
    #Send to device before passing to optimizer
    model_part.to(device)
    #Add model to watch list
    wandb.watch(model_part)
    model_part.eval()
   

#optimizer = SGD(all_params,lr=lr)



g_layers = [model_dict[key] for key in sorted(model_dict.keys()) if 'g_block' in key]
f_layers = [model_dict[key] for key in sorted(model_dict.keys()) if 'f_block' in key]

dataset = make_swiss_roll(n_samples=4000, noise=0.001, random_state=None)





rotator = Rotator().to(device)

optimizer = Adam(rotator.parameters(),lr=0.01)


original = dataset[0]

view_cloud_plotly(original)
rotated = rotate_mat(original - original.mean(axis=0),90,0,0) + original.mean(axis=0)

transformed = rotated + np.array([0,0,0])

view_cloud_plotly(np.concatenate([transformed,original],0))


transformed = torch.from_numpy(transformed).float().to(device)


for epoch in tqdm(range(n_epochs)):
    
    x = transformed

    optimizer.zero_grad()
    

    x = rotator(x).unsqueeze(0)

    
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


    if (epoch % 50 == 0) & (epoch !=0):
        with torch.no_grad():
            
            #print(rotator.translation)
            rgb1 = np.zeros_like(original)
            rgb2 = np.zeros_like(original)
            rgb2[:,0]= 1
            rgb = np.concatenate((rgb1,rgb2),axis=0)
            to_view = np.concatenate([rotator(transformed).cpu().numpy(),original],axis=0)
            view_cloud_plotly(to_view,rgb)
        #Save model
        # save_state_dict = {key:val.state_dict() for key,val in model_dict.items()}
        # save_state_dict['optimizer'] = optimizer.state_dict()
        # save_state_dict['scheduler'] = optimizer.state_dict()
        # final_save_path = save_model_path+ f"_{epoch}_" +wandb.run.name+".pt"
        # print(f"Saving model to {final_save_path}")
        # torch.save(save_state_dict,final_save_path)
    