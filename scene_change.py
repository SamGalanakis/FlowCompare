from torch_geometric.nn.pool import radius
from train import initialize_cross_flow,inner_loop_cross,load_cross_flow
import torch
import numpy as np
from torch_geometric.nn import fps
from utils import (
config_loader,
extract_area,
random_subsample,
save_las,get_all_voxel_centers,
)
from torch_cluster import grid_cluster
from dataloaders.dataset_utils import registration_pipeline



def change_map(cloud_0,cloud_1,model_path,change_func,out_path,batch_size,sample_size=2048,center = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cloud_0,cloud_1 = cloud_0.to(device),cloud_1.to(device)
    checkpoint_dict = torch.load(model_path)
    config = checkpoint_dict['config']
    models_dict = initialize_cross_flow(config,device,mode='train')
    models_dict = load_cross_flow(checkpoint_dict,models_dict)
    if center == None:
        center = cloud_0[:,:2].mean()
   
    
    cloud_0,cloud_1 = registration_pipeline([cloud_0,cloud_1],0.05,0.07)
    cloud_0 = cloud_0[extract_area(cloud_0,center,config['clearance'],shape='square'),...]
    cloud_1 = cloud_1[extract_area(cloud_1,center,config['clearance'],shape='square'),...]
    start,end = cloud_0.min(dim=0),cloud_0.max(dim=0)
    
    cluster_0 = grid_cluster(cloud_0[:,:3])
    voxel_centers = get_all_voxel_centers(start,end,config['final_voxel_size'])
    cluster_1 = grid_cluster(cloud_1[:,:3])




