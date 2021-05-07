from torch_geometric.nn.pool import radius
from train import initialize_cross_flow,inner_loop_cross,load_cross_flow
import torch
import numpy as np
from torch_geometric.nn import fps
from utils import (
grid_split,
config_loader,
extract_area,
knn_relator,
random_subsample,
save_las,
log_prob_to_color,
circle_cover,
co_unit_sphere
)



def change_map(cloud_0,cloud_1,clearance,config_path,model_path,change_func,out_path,batch_size,center = None,overlap=0,sample_size=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_loader(config_path)
    cloud_0,cloud_1 = cloud_0.to(device),cloud_1.to(device)
    models_dict = initialize_cross_flow(config,device,mode='train')
    checkpoint_dict = torch.load(model_path)
    models_dict = load_cross_flow(checkpoint_dict,models_dict)
    if center == None:
        center = cloud_0[:,:2].mean()
    cloud_0 = cloud_0[extract_area(cloud_0,center,clearance,shape='square'),...]
    cloud_1 = cloud_1[extract_area(cloud_1,center,clearance,shape='square'),...]
    grid = circle_cover(clearance,clearance,config['grid_square_size']/2,overlap=overlap,show=False)
    
    pairs = [[cloud_0[extract_area(cloud_0,point,radius),cloud_1[extract_area(cloud_1,point,radius)],...] for point in grid]




