import torch
from dataloaders import PostClassificationDataset
import wandb
import os
import pyro.distributions as dist
from utils import view_cloud_plotly,time_labeling,config_loader,oversample_cloud
import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
from tqdm import tqdm
from torch import nn
from models import DGCNN_cls
from torch.utils.data import DataLoader




post_config = config_loader('config/config_post_classification.yaml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#os.environ["WANDB_MODE"] = "dryrun"
config_path = r"config/config_conditional_cross.yaml"
wandb.init(project="flow_change",config = config_path)
config = wandb.config


one_up_path = os.path.dirname(__file__)
out_path = os.path.join(one_up_path,r"save/processed_dataset")


def collate_combine(batch):
    batch = [(time_labeling(x[0],x[1]),x[2]) for x in batch]
    combined_clouds = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    return torch.stack(combined_clouds),torch.stack(labels)


dataset = PostClassificationDataset('save/processed_dataset/super-pyramid-1528_probs_dataset.pt')
dataloader = DataLoader(dataset,shuffle=True,batch_size=post_config['batch_size'],collate_fn = collate_combine)

parameters = []
gcn = DGCNN_cls(input_dim = config['input_dim']+2,output_channels = len(dataset.class_int_dict),dropout = post_config['dropout'],k = post_config['k'], batchnorm=False).to(device)
parameters +=gcn.parameters()
optimizer = torch.optim.Adam(parameters, lr=post_config["lr"],weight_decay=post_config["weight_decay"]) 
criterion = nn.CrossEntropyLoss()

save_path = 'save/post_classification'
for epoch in range(post_config["n_epochs"]):
    loss_running_avg = 0
    for batch_ind,batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad(set_to_none=True)
        comb_cloud , label = batch
        comb_cloud,label = comb_cloud.to(device),label.to(device)
        net_out = gcn(comb_cloud)

        loss = criterion(net_out,label.squeeze())

        loss.backward()
        loss_item = loss.item()
        optimizer.step()
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({'loss':loss_item,'lr':current_lr})
        print(loss_item)
        loss_running_avg = (loss_running_avg*(batch_ind) + loss_item)/(batch_ind+1)
    
    torch.save(gcn,os.path.join(save_path,f"{wandb.run.name}_{epoch}_post_cls.pt"))
    wandb.log({'epoch':epoch,"loss_epoch":loss_running_avg})