import torch
import os
import numpy as np
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,extract_area,co_min_max,feature_assigner,Adamax,Early_stop
from tqdm import tqdm
from dataloaders import StraightChallengeFeatureLoader
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch import nn
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
one_up_path = os.path.dirname(__file__)
out_path = os.path.join(one_up_path,r"save/processed_dataset")

config_path = r"config/config_straight.yaml"

os.environ['WANDB_MODE'] = 'dryrun'
wandb.init(project="flow_change",config = config_path)
config = wandb.config
