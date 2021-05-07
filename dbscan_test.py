import cupoch as cph
import numpy as np
from utils import load_las,save_las,extract_area,random_subsample
import sys
import torch
from models.pct_utils import query_ball_point
import matplotlib.pyplot as plt







sys.path.append(r'C:\on_path')


cloud = torch.from_numpy(load_las(r'C:\Users\Sam\Desktop\0_WE1NZ71I.las')[:,:3]).cuda()
center = cloud[:,:2].mean(axis=0)
cloud = cloud[extract_area(cloud,center,10,'square'),...]
cloud = random_subsample(cloud,100000)

#indices = torch.randint(0,cloud.shape[0],10,device = 'cuda')


pcd = cph.geometry.PointCloud()
pcd.points = cph.utility.Vector3fVector(cloud.cpu().numpy())

labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=50, print_progress=True).cpu())
pass
max_label = labels.max()
cmap = plt.get_cmap("tab20")
colors = cmap(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = cph.utility.Vector3fVector(colors[:, :3])
cph.visualization.draw_geometries([pcd])
pass