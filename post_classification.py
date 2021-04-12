import torch
from conditional_cross_flow import initialize_cross_flow,load_cross_flow,inner_loop_cross,bits_per_dim
from dataloaders import PostClassificationDataset
import wandb
import os
import pyro.distributions as dist
from utils import view_cloud_plotly, bin_probs,time_labeling,config_loader
import pandas as pd
from visualize_change_map import visualize_change
import matplotlib.pyplot as plt
import  numpy as np
from tqdm import tqdm
from models import DGCNN_cls
from torch.utils.data import DataLoader





post_config = config_loader('config/config_post_classification.yaml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_MODE"] = "dryrun"
config_path = r"config/config_conditional_cross.yaml"
wandb.init(project="flow_change",config = config_path)
config = wandb.config


one_up_path = os.path.dirname(__file__)
out_path = os.path.join(one_up_path,r"save/processed_dataset")


def collate_combine(batch):
    batch = [(time_labeling(x[0],x[1]),x[2]) for x in batch]
    return batch


dataset = PostClassificationDataset('save/processed_dataset/probs_dataset_for_postclassification.pt')
dataloader = DataLoader(dataset,shuffle=True,batch_size=16,collate_fn = collate_combine)

model = DGCNN_cls(input_dim = config['input_dim'],output_channels = len(dataset.class_int_dict))

for batch_ind,batch in dataloader:
    pass