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
import laspy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

base_name = "double_block"
#a= load_las(r"save\change_maps\output.las")
grid_square_size = 4
downsample_number = 100000

points_1 = load_las(r"D:/data/cycloData/2016/0_5D4KVPBP.las")
points_2 = load_las(r"D:/data/cycloData/2020/0_WE1NZ71I.las")

# points_1 = pd.read_csv(r"D:\data\glacier_data\20150701_TLS_Hochebenkar_UTM32N\20150701_TLS_Hochebenkar_UTM32N.txt",sep="\t")
# points_1 = points_1[['X','Y','Z']].values
# points_2 = pd.read_csv(r"D:\data\glacier_data\20170719_TLS_Hochebenkar_UTM32N\20170719_TLS_Hochebenkar_UTM32N.txt",sep="\t")
# points_2 = points_2[['X','Y','Z']].values
clearance = 16
print(f"Starting grid, {(2*clearance/grid_square_size)**2} squares of area {grid_square_size}")
center = points_1[:,:2].mean(axis=0)
grid_1 = grid_split(points_1,grid_square_size,clearance= clearance,center = center)
grid_2 = grid_split(points_2,grid_square_size,clearance= clearance,center = center)

grid_1_subsampled = [random_subsample(x,downsample_number) for x in grid_1]
grid_2_subsampled = [random_subsample(x,downsample_number) for x in grid_2]


rgb_list = []
for index, (square_1,square_2) in enumerate(tqdm(zip(grid_1_subsampled,grid_2_subsampled))):
    log_probs_1, log_probs_2 = fit_flow(square_1,square_2)
    std = log_probs_1.std()
    rgb = np.zeros_like(log_probs_2)
    mask = log_probs_2<np.percentile(log_probs_1,0.01)
    rgb[mask] = np.abs(log_probs_2[mask])/std
    scaler = MinMaxScaler()
    rgb = scaler.fit_transform(rgb.reshape(-1,1)).reshape((-1,))
    rgb_list.append(rgb)


rgb_list = [knn_relator(grid_2[i],grid_2_subsampled[i],rgb_list[i]) for i in range(len(rgb_list))]
rgb_col = np.concatenate(rgb_list)
np.save(f"save//change_maps//{base_name}_rgb.npy",rgb_col)
points_2_for_file = np.concatenate(grid_2)
np.save(f"save//change_maps//{base_name}_coordinates.npy",points_2_for_file)
hdr = laspy.header.Header()

outfile = laspy.file.File(f"save//change_maps//change_map_{base_name}.las", mode="w", header=hdr)




outfile.define_new_dimension(
    name="change",
    data_type=10, # 
    description = "Change metric"
 )

outfile.writer.set_dimension('change',rgb_col)

allx = points_2_for_file[:,0] # Four Points
ally = points_2_for_file[:,1]
allz = points_2_for_file[:,2]


xmin = np.floor(np.min(allx))
ymin = np.floor(np.min(ally))
zmin = np.floor(np.min(allz))

outfile.header.offset = [xmin,ymin,zmin]
outfile.header.scale = [0.001,0.001,0.001]

outfile.x = allx
outfile.y = ally
outfile.z = allz


outfile.close()
view_cloud_plotly(points_2_for_file,rgb_col,colorscale='Hot')
#visualize_change(grid_1,grid_2,log_probs_list,log_probs_list)
