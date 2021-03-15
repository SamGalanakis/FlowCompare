from numpy.lib.function_base import extract
import torch
import os
import numpy as np
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,extract_area,co_min_max,feature_assigner,Adamax,Early_stop
from tqdm import tqdm
from dataloaders import StraightChallengeFeatureLoader,ChallengeDataset
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch import nn
import argparse
from utils import ground_remover

def log_prob_to_change(log_prob_0,log_prob_1,grads_1_given_0,config,percentile=1):
    std_0 = log_prob_0.std()
    perc = torch.Tensor([np.percentile(log_prob_0.cpu().numpy(),percentile)]).to(log_prob_0.device)
    change = torch.zeros_like(log_prob_1)
    mask = log_prob_1<=percentile
    change[mask] = (torch.abs(log_prob_1-perc)/std_0)[mask]
    abs_grads = torch.abs(grads_1_given_0)
    grads_sum_geom = abs_grads[:,:3].sum(axis=1)
    
    grads_sum_rgb = abs_grads[:,3:].sum(axis=1)
    eps= 1e-8
    if config['input_dim']>3:
        geom_rgb_ratio = grads_sum_geom/(grads_sum_rgb+eps)
    else:
        geom_rgb_ratio = grads_sum_geom
    return change,geom_rgb_ratio

def visualize_change(dataset,feature_dataset,index,colorscale='Hot',clearance=0.5,remove_ground =True):
    extract_0 = dataset[index][0]
    extract_1 = dataset[index][1]
    feature_dict = feature_dataset[index]
    if remove_ground:
        ground_mask_0 = ground_remover(extract_0)
        ground_mask_1 = ground_remover(extract_1)
    else:
        ground_mask_0 = torch.ones(extract_0.shape[0]).long()
        ground_mask_1 = torch.ones(extract_1.shape[0]).long()
    mean_ground_point_0 = extract_0[~ground_mask_0,:].mean(axis=0)
    mean_ground_point_1 = extract_1[~ground_mask_1,:].mean(axis=0)
    mean_log_prob_ground_point_0 = feature_dict['log_prob_0_given_0'].mean()
    mean_log_prob_ground_point_1 = feature_dict['log_prob_1_given_0'].mean()
    mean_grad_ground_point_1 = feature_dict['grads_1_given_0'].mean()
    extract_0 = extract_0[ground_mask_0,:]
    extract_1 = extract_1[ground_mask_1,:]
    feature_dict['log_prob_0_given_0'] = feature_dict['log_prob_0_given_0'][ground_mask_0]
    feature_dict['log_prob_1_given_0'] = feature_dict['log_prob_1_given_0'][ground_mask_1]
    feature_dict['grads_1_given_0'] = feature_dict['grads_1_given_0'][ground_mask_1]
    if extract_0.shape[0]==0:
        extract_0 = mean_ground_point_0.unsqueeze(0)
        feature_dict['log_prob_0_given_0'] = mean_log_prob_ground_point_0.unsqueeze(0)
    if extract_1.shape[0]==0:
        extract_1 = mean_ground_point_1.unsqueeze(0)
        feature_dict['log_prob_1_given_0'] = mean_log_prob_ground_point_1.unsqueeze(0)
        feature_dict['grads_1_given_0'] = mean_grad_ground_point_1.unsqueeze(0)

    extract_0[:,:3], extract_1[:,:3] = co_min_max(extract_0[:,:3],extract_1[:,:3])
    mask_0 = extract_area(extract_0,np.array([0.5,0.5]),clearance)
    mask_1 = extract_area(extract_1,np.array([0.5,0.5]),clearance)
    extract_0 = extract_0[mask_0,:]
    extract_1 = extract_1[mask_1,:]
    
    
    dataset.view(index)
    change, _ = log_prob_to_change(feature_dict['log_prob_0_given_0'][mask_0],feature_dict['log_prob_1_given_0'][mask_1],feature_dict['grads_1_given_0'][mask_1],config)
    return view_cloud_plotly(extract_1[:,:3],change,colorscale = colorscale)

def summary_stats(dataset,feature_dataset,index):
    feature_dict = feature_dataset[index]
    change_given_0, geom_rgb_ratio_given_0 = log_prob_to_change(feature_dict['log_prob_0_given_0'],feature_dict['log_prob_1_given_0'],feature_dict['grads_1_given_0'],config)
    change_given_1, geom_rgb_ratio_given_1 = log_prob_to_change(feature_dict['log_prob_1_given_1'],feature_dict['log_prob_0_given_1'],feature_dict['grads_0_given_1'],config)

    return change_given_0.mean(),geom_rgb_ratio_given_0.mean(),change_given_1.mean(),geom_rgb_ratio_given_1.mean()




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path,r"save/processed_dataset")

    config_path = r"config/config_straight.yaml"

    os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config
    dirs = [config['dir_challenge']+year for year in ["2016","2020"]]
    dataset = ChallengeDataset(config['dirs_challenge_csv'], dirs, out_path,subsample="fps",sample_size=config['sample_size'],preload=config['preload'],normalization=config['normalization'],subset=None,apply_normalization=False)
    feature_dataset = StraightChallengeFeatureLoader("save/processed_dataset/straight_features/")
    dataset.apply_normalization = False
    visualize_change(dataset,feature_dataset,10,clearance = 0.7)

