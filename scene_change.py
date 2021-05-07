from train import initialize_cross_flow,inner_loop_cross,load_cross_flow
import torch
import numpy as np
from utils import (
grid_split,
config_loader,
extract_area,
knn_relator,
random_subsample,
save_las,
log_prob_to_color
)



def change_map(cloud_0,cloud_1,config_path,model_path,change_func,out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_loader(config_path)
    cloud_0,cloud_1 = cloud_0.to(device),cloud_1.to(device)
    models_dict = initialize_cross_flow(config,device,mode='train')
    checkpoint_dict = torch.load(model_path)
    models_dict = load_cross_flow(checkpoint_dict,models_dict)
    



