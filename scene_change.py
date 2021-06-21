from torch_geometric.nn.pool import radius
from train import initialize_flow, inner_loop, load_flow
import torch
import numpy as np
from torch_geometric.nn import fps
from utils import (
grid_split,
config_loader,
extract_area,
random_subsample,
save_las,
log_prob_to_color,
circle_cover,
co_unit_sphere
)


def change_map(cloud_0, cloud_1, clearance, config_path, model_path, change_func, out_path, batch_size, center=None, overlap=0, sample_size=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_loader(config_path)
    checkpoint_dict = torch.load(model_path)
    config = checkpoint_dict['config']
    cloud_0, cloud_1 = cloud_0.to(device), cloud_1.to(device)
    models_dict = initialize_flow(config, device, mode='train')
    
    models_dict = load_flow(checkpoint_dict, models_dict)
    if center == None:
        center = cloud_0[:, :2].mean()
    cloud_0 = cloud_0[extract_area(
        cloud_0, center, clearance, shape='square'), ...]
    cloud_1 = cloud_1[extract_area(
        cloud_1, center, clearance, shape='square'), ...]

