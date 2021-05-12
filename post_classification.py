import torch
import os
import pyro.distributions as dist
from utils import view_cloud_plotly,time_labeling,config_loader,oversample_cloud,rotation_z
import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import math
import cupoch as cph
import matplotlib.pyplot as plt

change_data_path = ''
data_dict= torch.load(change_data_path)
def visualize_clusters():
    max_label = labels.max()
    cmap = plt.get_cmap("tab20")
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = cph.utility.Vector3fVector(colors[:, :3])
    cph.visualization.draw_geometries([pcd])

def dbscan_cluster(cloud,min_points,eps):
    pcd  = cph.geometry.PointCloud()
    pcd.points = cph.utility.Vector3fVector(cloud)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True).cpu())
    return labels

def get_clusters(data_dict,c_min,min_points,eps,):

    clusters = []
    for index, (cloud_0,cloud_1,label) in tqdm(enumerate(data_dict.items())):
        cloud_0 = cloud_0[cloud_0[:,-1]>c_min].cpu().numpy()
        cloud_1 = cloud_1[cloud_0[:,-1]>c_min].cpu().numpy()

        labels_0 = dbscan_cluster(cloud_0[:,:3],min_points=min_points,eps=eps)
        labels_1 = dbscan_cluster(cloud_1[:,:3],min_points=min_points,eps=eps)
        clusters.append([labels_0,labels_1])











