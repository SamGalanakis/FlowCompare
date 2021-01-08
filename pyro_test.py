import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from utils import load_las, random_subsample,view_cloud_plotly
from pyro.nn import DenseNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_samples=1000
input_dim=3
sign = load_las("data\sign_1.las")[:,0:3]
sign_2 = load_las("data\sign_2.las")[:,0:3]
scaler = StandardScaler()
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


flow_blocks = [flow_block(input_dim,permutations,count_bins,split_dims,device) for x in range(n_blocks)]

parameters = []
transformations = []
for flow_block in flow_blocks:
    parameters.extend(flow_block.parameters)
    transformations.extend(flow_block.transformations)


flow_dist = dist.TransformedDistribution(base_dist, transformations)




steps = 30000

optimizer = torch.optim.Adam(parameters, lr=5e-3)

for step in range(steps+1):
    X = random_subsample(sign_scaled,n_samples)
    dataset = torch.tensor(X, dtype=torch.float).to(device)
    optimizer.zero_grad()
    loss = -flow_dist.log_prob(dataset).mean()
    loss.backward()
    optimizer.step()
    flow_dist.clear_cache()

    if step % 500 == 0:
        print('step: {}, loss: {}'.format(step, loss.item()))
        if step>0:
            with torch.no_grad():
                sample = flow_dist.sample([10000]).cpu()
                fixed_sample = scaler.inverse_transform(sample.numpy())
                
                view_cloud_plotly(fixed_sample[:,0:3],show=False).write_html(f'save//graphs//sample_{step}.html')
                rgb  = flow_dist.log_prob(torch.tensor(sign_scaled, dtype=torch.float).to(device)).detach().cpu().numpy()
                view_cloud_plotly(sign,rgb,colorscale='Hot',show_scale=True,show=False).write_html(f'save//graphs//t1_{step}.html')
                rgb = flow_dist.log_prob(torch.tensor(sign_2_scaled, dtype=torch.float).to(device)).detach().cpu().numpy()
                view_cloud_plotly(sign_2,rgb,colorscale='Hot',show_scale=True,show=False).write_html(f'save//graphs//t2_{step}.html')
            

pass