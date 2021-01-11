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
from utils import load_las, random_subsample,view_cloud_plotly, knn_relator
from pyro.nn import DenseNN
from visualize_change_map import visualize_change
from flow_fitter import fit_flow


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_samples=100
input_dim=3
sign = load_las("data\sign_1.las")[:,0:3]
sign_2 = load_las("data\sign_2.las")[:,0:3]
log_prob_2,log_prob_1 = fit_flow(sign,sign_2)
scaler = MinMaxScaler()
scaler.fit(np.concatenate((sign,sign_2),axis=0))

sign_scaled = scaler.transform(sign)
sign_2_scaled = scaler.transform(sign_2)


base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))
count_bins = 16
param_dims = lambda split_dimension: [(input_dim - split_dimension) * count_bins,
(input_dim - split_dimension) * count_bins,
(input_dim - split_dimension) * (count_bins - 1),
(input_dim - split_dimension) * count_bins]

split_dims = [2]*3

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

flow_blocks = torch.load('save//pyro_saves//flow_blocks_3000.pcl')

parameters = []
transformations = []
for flow_block_instance in flow_blocks:
    parameters.extend(flow_block_instance.parameters)
    transformations.extend(flow_block_instance.transformations)


flow_dist = dist.TransformedDistribution(base_dist, transformations)


rgb1= flow_dist.log_prob(torch.tensor(sign_scaled, dtype=torch.float).to(device)).detach().cpu().numpy()
rgb2 = flow_dist.log_prob(torch.tensor(sign_2_scaled, dtype=torch.float).to(device)).detach().cpu().numpy()
view_cloud_plotly(sign_2,np.abs(rgb2-rgb1.mean())**2,colorscale='Hot')
visualize_change([sign],[sign_2],[rgb1],[rgb2])


# with torch.no_grad():
#     sample = flow_dist.sample([10000]).cpu()
#     fixed_sample = scaler.inverse_transform(sample.numpy())
    
#     #view_cloud_plotly(fixed_sample[:,0:3],show=True)
#     rgb  = flow_dist.log_prob(torch.tensor(sign_scaled, dtype=torch.float).to(device)).detach().cpu().numpy()
#     view_cloud_plotly(sign,rgb,colorscale='Hot',show_scale=True,show=True)
#     rgb = flow_dist.log_prob(torch.tensor(sign_2_scaled, dtype=torch.float).to(device)).detach().cpu().numpy()
#     view_cloud_plotly(sign_2,rgb,colorscale='Hot',show_scale=True,show=True)

