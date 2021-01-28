import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,Early_stop,grid_split
from pyro.nn import DenseNN
from torch.utils.data import Dataset, DataLoader
from models.point_encoders import PointnetEncoder
from itertools import permutations 
from tqdm import tqdm
from models.pytorch_geometric_pointnet2 import Pointnet2
from torch_geometric.nn import fps
from dataloaders.ConditionalDataGrid import ConditionalDataGrid
dirs = [r'D:\data\cycloData\multi_scan\2018',r'D:\data\cycloData\multi_scan\2020']


def my_collate(batch):
    extract_0 = [item[0] for item in batch]
    extract_1 = [item[1] for item in batch]
    batch_id_0 = [torch.ones(x.shape[0],dtype=torch.long)*index for index,x in enumerate(extract_0)]
    batch_id_1 = [torch.ones(x.shape[0],dtype=torch.long)*index for index,x in enumerate(extract_1)]
    extract_0 = torch.cat(extract_0)
    extract_1 = torch.cat(extract_1)
    batch_id_0 = torch.cat(batch_id_0)
    batch_id_1 = torch.cat(batch_id_1)
    return [extract_0, batch_id_0, extract_1,batch_id_1]

dataset=ConditionalDataGrid(dirs,out_path="save//processed_dataset",preload=True,subsample='random')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(dataset,shuffle=True,batch_size=3,num_workers=0,prefetch_factor=2,collate_fn=my_collate)
Pointnet2 = Pointnet2(feature_dim=3)

for extract_0, batch_id_0, extract_1,batch_id_1 in dataloader:
    x = extract_0[:,3:]
    Pointnet2(x,extract_0[:,:3],batch_id_0)
    pass

input_dim = 6

base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))
count_bins = 16
param_dims = lambda split_dimension: [(input_dim - split_dimension) * count_bins,
(input_dim - split_dimension) * count_bins,
(input_dim - split_dimension) * (count_bins - 1),
(input_dim - split_dimension) * count_bins]

split_dims = [2]*3
patience = 50
n_blocks = 2
permutations =  [[1,0,2],[2,0,1],[2,1,0]]


class conditional_flow_block:
    def __init__(self,input_dim,permutations,count_bins,split_dims,device):
        self.transformations = []
        self.parameters =[]
        param_dims = lambda split_dimension: [(input_dim - split_dimension) * count_bins,
        (input_dim - split_dimension) * count_bins,
        (input_dim - split_dimension) * (count_bins - 1),
        (input_dim - split_dimension) * count_bins]
        for i, permutation in enumerate(permutations):
            hypernet =  DenseNN(split_dims[i], [10*input_dim], param_dims(split_dims[i]))
            spline = T.ConditionalSpline(input_dim = input_dim, split_dim = split_dims[i] , count_bins=count_bins,nn=hypernet)
            spline = spline.to(device)
            self.parameters += spline.parameters()
            self.transformations.append(spline)
            self.transformations.append(T.permute(input_dim,torch.LongTensor(permutations[i]).to(device),dim=-1))
    def save(self,path):
        torch.save(self,path)

flow_blocks = [flow_block(input_dim,permutations,count_bins,split_dims,device) for x in range(n_blocks)]

parameters = []
transformations = []
for flow_block_instance in flow_blocks:
    parameters.extend(flow_block_instance.parameters)
    transformations.extend(flow_block_instance.transformations)


flow_dist = dist.TransformedDistribution(base_dist, transformations)




steps = 3000
early_stop_margin=0.01
optimizer = torch.optim.AdamW(parameters, lr=5e-3) #was 5e-3

early_stop = Early_stop(patience=patience,min_perc_improvement=torch.tensor(early_stop_margin))

for step in range(steps+1):
    X = random_subsample(points1_scaled,n_samples)
    #dataset = torch.tensor(X, dtype=torch.float).to(device)
    dataset=X
    dataset += torch.randn_like(dataset)*0.01
    optimizer.zero_grad()
    loss = -flow_dist.log_prob(dataset).mean()
    
    
    
    
    last_loss = loss
    loss.backward()
    optimizer.step()
    flow_dist.clear_cache()

    stop_training = early_stop.log(loss.detach().cpu())

    if stop_training:
        print(f"Ran out of patience at step: {step}, Cur_loss: {loss}, Best: {early_stop.best_loss}")
        
        break

    
    

log_probs_1 = flow_dist.log_prob(points1_scaled).detach().cpu().numpy()
log_probs_2 = flow_dist.log_prob(points2_scaled).detach().cpu().numpy()


    
