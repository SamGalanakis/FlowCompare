
import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,knn_relator,save_las,extract_area,co_min_max
from pyro.nn import DenseNN
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.pytorch_geometric_pointnet2 import Pointnet2
from torch_geometric.nn import fps
from dataloaders.ConditionalDataGrid import ConditionalDataGrid
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import torch.distributed as distributed
from torch_geometric.nn import DataParallel as geomDataParallel
from models.pytorch_geometric_pointnet2 import Pointnet2
from dataloaders.ConditionalDataGrid import ConditionalDataGrid
import laspy




device = 'cuda'



def ready_module(transform):
    try:
        transform = transform.to(device) 
    except: 
        pass
   
    try:
        transform = transform.train()
    except: 
        pass
    return transform


grid_square_size = 4
context_dim =32
input_dim = 3
model_dict_path = r"save\conditional_flow_compare\0_16068_model_dict.pt"
model_dict = torch.load(model_dict_path)
transformations = model_dict['flow_transformations']
transformations = [ready_module(x) for x in transformations]


Pointnet2 = Pointnet2(feature_dim=input_dim-3,out_dim=context_dim).eval()
Pointnet2.load_state_dict(model_dict['encoder_dict'])
Pointnet2 = Pointnet2.to(device)
base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))
flow_dist = dist.ConditionalTransformedDistribution(base_dist, transformations)

points_0 = load_las(r"D:\data\cycloData\2016\0_5D4KVPBP.las")[:,:input_dim]
points_1 = load_las(r"D:\data\cycloData\2020\0_WE1NZ71I.las")[:,:input_dim]
sign_point = np.array([86967.46,439138.8])

sign_0 = extract_area(points_0,sign_point,1.5,'square')
sign_0 = torch.from_numpy(sign_0.astype(dtype=np.float32)).to(device)

sign_1 = extract_area(points_1,sign_point,1.5,'square')
sign_1= torch.from_numpy(sign_1.astype(dtype=np.float32)).to(device)
sign_0, sign_1 = co_min_max(sign_0,sign_1)
batch_id_0 = torch.zeros(sign_0.shape[0],dtype=torch.long).to(device)
encoding = Pointnet2(None,sign_0[:,:3],batch_id_0)
encoded = flow_dist.condition(encoding.unsqueeze(-2))
samples = encoded.sample([2000]).squeeze()
samples = random_subsample(samples,10000)
view_cloud_plotly(samples[:,:3])

clearance = 12
print(f"Starting grid, {(2*clearance/grid_square_size)**2} squares of area {grid_square_size}")
center = points_1[:,:2].mean(axis=0)
grid_1 = grid_split(points_1,grid_square_size,clearance= clearance,center = center)
grid_2 = grid_split(points_2,grid_square_size,clearance= clearance,center = center)

grid_1_subsampled = [random_subsample(x,downsample_number) for x in grid_1]
grid_2_subsampled = [random_subsample(x,downsample_number) for x in grid_2]


rgb_list = []

for index, (tensor_0,tensor_1) in enumerate(tqdm(zip(grid_1_subsampled,grid_2_subsampled))):
    tensor_0 = torch.from_numpy(tensor_0.astype(np.float32))
    tensor_1 = torch.from_numpy(tensor_1.astype(np.float32))
    overall_max = torch.max(tensor_0[:,:3].max(axis=0)[0],tensor_1[:,:3].max(axis=0)[0])
    overall_min = torch.min(tensor_0[:,:3].min(axis=0)[0],tensor_1[:,:3].min(axis=0)[0])
    tensor_0[:,:3] = (tensor_0[:,:3] - overall_min)/(overall_max-overall_min)
    tensor_1[:,:3] = (tensor_1[:,:3] - overall_min)/(overall_max-overall_min)

    batch_id_0 = (torch.zeros(tensor_0.shape[0],dtype=torch.long)).to(device)
    batch_id_1 = (torch.zeros(tensor_1.shape[0],dtype=torch.long)).to(device)
    tensor_0 = tensor_0.to(device)
    tensor_1 = tensor_1.to(device)
    encodings_0 = Pointnet2(tensor_0[:,3:],tensor_0[:,:3],batch_id_0) 
    encodings_1 = Pointnet2(tensor_1[:,3:],tensor_1[:,:3],batch_id_1)
    reconstruction_given_0 = flow_dist.condition(encodings_0).sample([2000]).cpu().numpy().squeeze()
    save_las(reconstruction_given_0[:,:3],f"/mnt/cm-nas03/synch/students/sam/change_maps/{index}_reconstruction.las",reconstruction_given_0[:,3:])
    save_las(tensor_0[:,:3].cpu().detach().numpy(),f"/mnt/cm-nas03/synch/students/sam/change_maps/{index}_original.las",tensor_0[:,3:].cpu().detach().numpy())
    prob_0_given_1 = flow_dist.condition(encodings_1).log_prob(tensor_0).cpu().detach().numpy().squeeze()
    prob_1_given_1 = flow_dist.condition(encodings_1).log_prob(tensor_1).cpu().detach().numpy().squeeze()
    prob_1_given_0 = flow_dist.condition(encodings_0).log_prob(tensor_1).cpu().detach().numpy().squeeze()
    prob_0_given_0 = flow_dist.condition(encodings_0).log_prob(tensor_0).cpu().detach().numpy().squeeze()
    
    std_1_given_1 = prob_1_given_1.std()
    std_0_given_0 = prob_0_given_0.std()
    rgb = np.zeros_like(prob_1_given_0)
    percentile = np.percentile(prob_0_given_0,0.01)
    mask = prob_1_given_0 < percentile
    rgb[mask] = np.abs(prob_1_given_0[mask]-percentile)/std_1_given_1
    rgb = np.abs(prob_1_given_0-percentile)**2
    scaler = MinMaxScaler()
    rgb = scaler.fit_transform(rgb.reshape(-1,1)).reshape((-1,))
    rgb_list.append(rgb)

base_name = 'change_map'
rgb_list = [knn_relator(grid_2[i],grid_2_subsampled[i],rgb_list[i]) for i in range(len(rgb_list))]
rgb_col = np.concatenate(rgb_list)

points_2_for_file = np.concatenate(grid_2)

save_las(points_2_for_file[:,:3],f"/mnt/cm-nas03/synch/students/sam/change_maps/change_map_{base_name}.las",points_2_for_file[:,3:],rgb_col)