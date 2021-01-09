import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly
from pyro.nn import DenseNN


def fit_flow(points1,points2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_samples=100
    input_dim=3
    points1 = points1[:,:input_dim]
    points2 = points2[:,:input_dim] 
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((points1,points2),axis=0))

    points1_scaled = scaler.transform(points1)
    points2_scaled = scaler.transform(points2)
    points2_scaled = torch.tensor(points2_scaled, dtype=torch.float).to(device)

    base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))
    count_bins = 16
    param_dims = lambda split_dimension: [(input_dim - split_dimension) * count_bins,
    (input_dim - split_dimension) * count_bins,
    (input_dim - split_dimension) * (count_bins - 1),
    (input_dim - split_dimension) * count_bins]

    split_dims = [2]*3
    patience = 10
    not_improved_count = 0
    n_blocks = 1
    permutations =  [[1,0,2],[2,0,1],[2,1,0]]


    class flow_block:
        def __init__(self,input_dim,permutations,count_bins,split_dims,device):
            self.transformations = []
            self.parameters =[]
            param_dims = lambda split_dimension: [(input_dim - split_dimension) * count_bins,
            (input_dim - split_dimension) * count_bins,
            (input_dim - split_dimension) * (count_bins - 1),
            (input_dim - split_dimension) * count_bins]
            for i, permutation in enumerate(permutations):
                hypernet =  DenseNN(split_dims[i], [10*input_dim], param_dims(split_dims[i]))
                spline = T.SplineCoupling(input_dim = input_dim, split_dim = split_dims[i] , count_bins=count_bins,hypernet=hypernet)
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
    optimizer = torch.optim.Adam(parameters, lr=5e-3)
    min_loss = torch.tensor(1e+8)


    for step in range(steps+1):
        X = random_subsample(points1_scaled,n_samples)
        dataset = torch.tensor(X, dtype=torch.float).to(device)
        dataset += torch.randn_like(dataset)*0.01
        optimizer.zero_grad()
        loss = -flow_dist.log_prob(dataset).mean()
        

        
       
        last_loss = loss
        loss.backward()
        optimizer.step()
        flow_dist.clear_cache()

        if loss< -torch.abs(min_loss)*early_stop_margin + min_loss:
            min_loss = loss
            not_improved_count=0
            best_log_probs = flow_dist.log_prob(points2_scaled).detach().cpu().numpy()
        else:
            not_improved_count+=1
            if not_improved_count>patience:
                print(f"Ran out of patience at step: {step}")
                break
        
        

    
    return best_log_probs

        
    