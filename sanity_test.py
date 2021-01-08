from sklearn.datasets import make_swiss_roll
import numpy as np
from models.model_init import model_init
import torch
import wandb
from tqdm import tqdm
from torch.optim import Adam, SGD
from utils import (
load_las,
compare_clouds,
extract_area,
random_subsample,
compare_clouds,
view_cloud_plotly,
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

prior_z = torch.distributions.MultivariateNormal(
    torch.zeros(3), torch.eye(3)
)
x= prior_z.sample(sample_shape=(1,100000))
view_cloud_plotly(x.squeeze().numpy(),torch.exp(prior_z.log_prob(x)).numpy()[0],show_scale=True,colorscale='Hot')
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
view_dataset = make_swiss_roll(n_samples=6000, noise=0.0, random_state=None)
view_cloud_plotly(view_dataset[0],np.zeros_like(view_dataset[0]))


g_layers = [model_dict[key] for key in sorted(model_dict.keys()) if 'g_block' in key]
f_layers = [model_dict[key] for key in sorted(model_dict.keys()) if 'f_block' in key]

dataset = make_swiss_roll(n_samples=100000, noise=0.001, random_state=None)

dataset = torch.from_numpy(dataset[0]).float().to(device)

batch_size=100
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  

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


        if (epoch % 199 == 0) & (epoch !=0):
            with torch.no_grad():
                x = prior_z.sample(sample_shape = (1,sample_size*10))
                rgb  = torch.exp(prior_z.log_prob(x.squeeze())).numpy()
                x = x.to(device)
                for g_layer in g_layers[::-1]:
                    x = g_layer.inverse(x)
                x = x.cpu().squeeze()
                view_cloud_plotly(x.numpy(),rgb,show_scale=True,colorscale='Hot')
            #Save model
            save_state_dict = {key:val.state_dict() for key,val in model_dict.items()}
            save_state_dict['optimizer'] = optimizer.state_dict()
            save_state_dict['scheduler'] = optimizer.state_dict()
            final_save_path = save_model_path+ f"_{epoch}_" +wandb.run.name+".pt"
            print(f"Saving model to {final_save_path}")
            torch.save(save_state_dict,final_save_path)
print(prof)