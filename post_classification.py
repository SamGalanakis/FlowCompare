import torch
from conditional_cross_flow import initialize_cross_flow,load_cross_flow,inner_loop_cross,bits_per_dim,time_labeling
from dataloaders import ConditionalDataGrid, ShapeNetLoader,ChallengeDataset
import wandb
import os
import pyro.distributions as dist
from utils import view_cloud_plotly, bin_probs
import pandas as pd
from visualize_change_map import visualize_change
import matplotlib.pyplot as plt
import  numpy as np
from tqdm import tqdm
from models import DGCNN
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_MODE"] = "dryrun"
config_path = r"config/config_conditional_cross.yaml"
wandb.init(project="flow_change",config = config_path)
config = wandb.config
load_path =   r"save/conditional_flow_compare/super-pyramid-1528_372_model_dict.pt"            #r"save/conditional_flow_compare/likely-eon-1555_139_model_dict.pt"
save_dict = torch.load(load_path)
model_dict = initialize_cross_flow(config,device,mode='test')
model_dict = load_cross_flow(save_dict,model_dict)
mode = 'test'


one_up_path = os.path.dirname(__file__)
out_path = os.path.join(one_up_path,r"save/processed_dataset")

dirs_challenge_csv = 'save/2016-2020-train/'.replace('train',mode)
dirs = ['save/challenge_data/Shrec_change_detection_dataset_public/'+year for year in ["2016","2020"]]
dataset = ChallengeDataset(dirs_challenge_csv, dirs, out_path,subsample="fps",sample_size=2000,preload=False,normalization=config['normalization'],subset=None,radius=int(config['grid_square_size']/2),remove_ground=False,mode = mode,hsv=False)

def collate_postclassification(batch):
    extract_0 = [item[0][:config['sample_size'],:config["input_dim"]] for item in batch]
    extract_1 = [item[1][:config['sample_size'],:config["input_dim"]] for item in batch]
    

    extract_1 = torch.stack(extract_1)
    return [extract_0, extract_1]

dataloader = DataLoader(dataset,shuffle=True,batch_size=10,collate_fn = collate_postclassification)

model = DGCNN(output_channels = len(dataset.class_int_dict),)

for batch_ind,batch in dataloader:
    pass