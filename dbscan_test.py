import cupoch as cph
import numpy as np
from utils import load_las,save_las,extract_area,random_subsample,grid_split,view_cloud_plotly,view_cloud_o3d
import sys
import torch
from models.pct_utils import query_ball_point
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.io as pio
import webbrowser

pio.renderers.default = "browser"


sys.path.append(r'C:\on_path')


cloud = torch.from_numpy(load_las(r'/media/raid/sam/ams_dataset/5D74EOHE.laz')).cuda().float()
center = cloud[:,:2].mean(axis=0)
# cloud = cloud[extract_area(cloud,center,10,'square'),...]
# grid = grid_split(cloud,1,center=center,clearance =10)
# cloud = random_subsample(cloud,10000).cpu().numpy()
# #view_cloud_o3d(cloud[:,:3],cloud[:,3:])

# fig = view_cloud_plotly(grid[30][:,:3],show=False)
# fig.show()
pcd = cph.geometry.PointCloud()
pcd.points = cph.utility.Vector3fVector(cloud.cpu().numpy()[:,:3])
pcd.colors = cph.utility.Vector3fVector(cloud.cpu().numpy()[:,3:])
pcd = pcd.voxel_down_sample(0.07)
save_las(np.asarray(pcd.points.cpu()),'temp.las',np.asarray(pcd.colors.cpu()))
# cph.visualization.draw_geometries([pcd])
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=50, print_progress=True).cpu())
pass
max_label = labels.max()
cmap = plt.get_cmap("tab20")
colors = cmap(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = cph.utility.Vector3fVector(colors[:, :3])
cph.visualization.draw_geometries([pcd])
pass