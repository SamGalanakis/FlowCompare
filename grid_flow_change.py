import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import datasets
from utils import load_las, random_subsample,view_cloud_plotly, grid_split, knn_relator
from pyro.nn import DenseNN
from visualize_change_map import visualize_change
from flow_fitter import fit_flow
from tqdm import tqdm 
import pandas as pd

grid_square_size = 4
downsample_number = 100000

# points_1 = load_las(r"D:/data/cycloData/2016/0_5D4KVPBP.las")
# points_2 = load_las(r"D:/data/cycloData/2020/0_WE1NZ71I.las")

points_1 = pd.read_csv(r"D:\data\glacier_data\20150701_TLS_Hochebenkar_UTM32N\20150701_TLS_Hochebenkar_UTM32N.txt",sep="\t")
points_1 = points_1[['X','Y','Z']].values
points_2 = pd.read_csv(r"D:\data\glacier_data\20170719_TLS_Hochebenkar_UTM32N\20170719_TLS_Hochebenkar_UTM32N.txt",sep="\t")
points_2 = points_2[['X','Y','Z']].values
grid_1 = grid_split(points_1,grid_square_size,clearance= 5)
grid_2 = grid_split(points_2,grid_square_size,clearance= 5)

grid_1 = [random_subsample(x,downsample_number) for x in grid_1]
grid_2 = [random_subsample(x,downsample_number) for x in grid_2]

log_probs_list = []
for points_1,points_2 in tqdm(zip(grid_1,grid_2)):
    log_probs = fit_flow(points_1,points_2)
    log_probs_list.append(log_probs)


visualize_change(grid_1,grid_2,log_probs_list,log_probs_list)
