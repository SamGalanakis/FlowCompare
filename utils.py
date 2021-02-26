import torch
import numpy as np
import time
from laspy.file import File
import pandas as pd
import plotly.graph_objects as go
from matplotlib import widgets
from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib.artist as artist
import matplotlib.image as image
from scipy.spatial.transform import Rotation 
from sklearn.neighbors import NearestNeighbors
import os
import math
import laspy
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.data import Data,Batch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
#Losses from original repo

eps = 1e-8


def loss_fun(z, z_ldetJ, prior_z, e, e_ldetJ, prior_e):
    ll_z = prior_z.log_prob(z.cpu()).to(z.device) + z_ldetJ
    ll_e = prior_e.log_prob(e.cpu()).to(e.device) + e_ldetJ
    return -torch.mean(ll_z), -torch.mean(ll_e)


def loss_fun_ret(z, z_ldetJ, prior_z):
    ll_z = prior_z.log_prob(z.cpu()).to(z.device) + z_ldetJ
    return -torch.mean(ll_z)

def knn_relator(points,points_subsampled,feature,n_neighbors=1):
    if points.shape[0] == points_subsampled.shape[0]:
        return feature
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points_subsampled)
    indices = nbrs.kneighbors(points,return_distance=False)
    feature_original = feature[indices].mean(axis=1)

        

    
    return feature_original

    

def load_las(path,extra_dim_list=None,scale_colors = True):
    input_las = File(path, mode='r')
    point_records = input_las.points.copy()
    las_scaleX = input_las.header.scale[0]
    las_offsetX = input_las.header.offset[0]
    las_scaleY = input_las.header.scale[1]
    las_offsetY = input_las.header.offset[1]
    las_scaleZ = input_las.header.scale[2]
    las_offsetZ = input_las.header.offset[2]

    # calculating coordinates
    if scale_colors:
        color_div=65536
    else:
        color_div=1
    p_X = np.array((point_records['point']['X'] * las_scaleX) + las_offsetX)
    p_Y = np.array((point_records['point']['Y'] * las_scaleY) + las_offsetY)
    p_Z = np.array((point_records['point']['Z'] * las_scaleZ) + las_offsetZ)
    try:
        points = np.vstack((p_X,p_Y,p_Z,input_las.red/color_div,input_las.green/color_div,input_las.blue/color_div)).T
    except:
        pass
    
    return points

def view_cloud_plotly(points,rgb=None,fig=None,point_size=2,show=True,axes=False,show_scale=False,colorscale=None):
    if  isinstance(points,torch.Tensor):
        points = points.cpu()
        points = points.numpy()
    if  isinstance(rgb,torch.Tensor):
        
        rgb = rgb.cpu().numpy()
    if  rgb is None:
        rgb = np.zeros_like(points)
    else:
        rgb = np.rint(np.divide(rgb,rgb.max(axis=0))*255).astype(np.uint8)
    if fig==None:
        fig = go.Figure()
    fig.add_scatter3d(
        x=points[:,0], 
        y=points[:,1], 
        z=points[:,2], 
        marker=dict(
        size=point_size,
        color=rgb,  
        colorscale=colorscale,
        showscale=show_scale,
        opacity=1
    ), 
        
        opacity=1, 
        mode='markers',
        
    )


    if not axes:
            fig.update_layout(
            scene=dict(
            xaxis=dict(showticklabels=False,visible= False),
            yaxis=dict(showticklabels=False,visible= False),
            zaxis=dict(showticklabels=False, visible= False),
            )
)

    if show:
        fig.show()
    return fig
    
  
    
    

def extract_area(full_cloud,center,clearance,shape= 'cylinder'):
    if shape == 'square':
        x_mask = ((center[0]+clearance)>full_cloud[:,0]) &   (full_cloud[:,0] >(center[0]-clearance))
        y_mask = ((center[1]+clearance)>full_cloud[:,1]) &   (full_cloud[:,1] >(center[1]-clearance))
        mask = x_mask & y_mask
    elif shape == 'cylinder':
        mask = np.linalg.norm(full_cloud[:,:2]-center,axis=1) <  clearance
    return full_cloud[mask]

def grid_split(points,grid_size,center = False,clearance = 20):
    if isinstance(center,bool):
        center = points[:,:2].mean(axis=0)
    
    
    center_x = center[0]
    center_y= center[1]
    
    x = np.arange(center_x-clearance, center_x+clearance, grid_size)
    y = np.arange(center_y-clearance, center_y+clearance, grid_size)
    grid_list = []
    for x_val in x:
        mask_x = np.logical_and(points[:,0]>x_val,points[:,0]<x_val+grid_size)
        x_strip = points[mask_x]
        for y_val in y:
            mask_y = np.logical_and(x_strip[:,1]>y_val,x_strip[:,1]<y_val+grid_size)
            tile = x_strip[mask_y]
            grid_list.append(tile)
    return grid_list
def circle_split(points,circle_radius,center = False,clearance = 20):
    if isinstance(center,bool):
        center = points[:,:2].mean(axis=0)
    
    
    center_x = center[0]
    center_y= center[1]
    x = np.arange(center_x-clearance, center_x+clearance, circle_radius) 
    y = np.arange(center_y-clearance, center_y+clearance, circle_radius) 
   
    circles_list = []
    for x_val in x:
        for y_val in y:
            tile = points[np.linalg.norm(points[:,:2]-np.array([x_val,y_val]),axis=1)<circle_radius]
            circles_list.append(tile)
    return circles_list


def random_subsample(points,n_samples):
    if points.shape[0]==0:
        print('No points found at this center replacing with dummy')
        points = np.zeros((1,points.shape[1]))
    #No point sampling if already 
    if n_samples < points.shape[0]:
        random_indices = np.random.choice(points.shape[0],n_samples, replace=False)
        points = points[random_indices,:]
    
    
    
    
    return points



def rotate_mat(points,x,y,z):
    rx = Rotation.from_euler('x', x, degrees=True)
    ry = Rotation.from_euler('y', y, degrees=True)
    rz = Rotation.from_euler('z', z, degrees=True)
    full_rot = (rx*ry*rz).as_matrix()
    return (np.matmul(full_rot,points.T)).T

    


    
class Early_stop:
    def __init__(self,patience=50,min_perc_improvement=0):
        self.patience = patience
        self.min_perc_improvement = min_perc_improvement
        self.not_improved = 0
        self.best_loss = torch.tensor(1e+8)
        self.last_loss = self.best_loss
        self.step = -1
        self.loss_hist = []
    def log(self,loss):
        self.loss_hist.append(loss)
        self.step+=1
        #Check if improvement by sufficient margin
        if (torch.abs(self.last_loss-loss) > torch.abs(self.min_perc_improvement*self.last_loss)) & (self.last_loss<loss):
            self.not_improved = 0
        else:
            self.not_improved +=1
        
        #Keep track of best loss
        if loss < self.best_loss:
            self.best_loss = loss
        #Set last_loss
        self.last_loss = loss
        if self.not_improved > self.patience:
            stop_training = True
        else:
            stop_training = False
        return stop_training


def save_las(pos,path,rgb=None,extra_feature=None,feature_name='Change'):
    hdr = laspy.header.Header(point_format=2)

    
    outfile = laspy.file.File(path, mode="w", header=hdr)
    if not isinstance( extra_feature,type(None)):
        outfile.define_new_dimension(
            name=feature_name,
            data_type=10, # 
            description = "Change metric"
        )

        outfile.writer.set_dimension('change',extra_feature)

    allx = pos[:,0] # Four Points
    ally = pos[:,1]
    allz = pos[:,2]


    xmin = np.floor(np.min(allx))
    ymin = np.floor(np.min(ally))
    zmin = np.floor(np.min(allz))

    outfile.header.offset = [xmin,ymin,zmin]
    outfile.header.scale = [0.001,0.001,0.001]

    outfile.x = allx
    outfile.y = ally
    outfile.z = allz

    if not isinstance( rgb,type(None)):
        if rgb.shape[-1]==6:
            outfile.red = rgb[:,0]
            outfile.green = rgb[:,1]
            outfile.blue = rgb[:,2]
        else:
            rgb= None
        


    outfile.close()

def co_min_max(tensor_0,tensor_1):
    overall_max = torch.max(tensor_0[:,:3].max(axis=0)[0],tensor_1[:,:3].max(axis=0)[0])
    overall_min = torch.min(tensor_0[:,:3].min(axis=0)[0],tensor_1[:,:3].min(axis=0)[0])
    denominator = overall_max-overall_min  + eps
    tensor_0[:,:3] = (tensor_0[:,:3] - overall_min)/denominator
    tensor_1[:,:3] = (tensor_1[:,:3] - overall_min)/denominator

    if (tensor_0.isnan().any() or tensor_1.isnan().any()).item():
            raise Exception("Nan in __item__!")


    return tensor_0,tensor_1
    

def expm(x):
    return torch.matrix_exp(x)

def feature_assigner(x,input_dim):
    return None if input_dim==3 else x[:,3:]

def collate_voxel(batch,voxel_size,input_dim,start=0,end=1):
    
    extract_0 = [item[0][:,:input_dim] for item in batch]
    extract_1 = [item[1][:,:input_dim] for item in batch]
    data_list_0 = [Data(x=feature_assigner(x,input_dim),pos=x[:,:3]) for x in extract_0]
    data_list_1 = [Data(x=feature_assigner(x,input_dim),pos=x[:,:3]) for x in extract_1]
    batch_0 = Batch.from_data_list(data_list_0)
    batch_1 = Batch.from_data_list(data_list_1)

    batch_0_voxels = voxel_grid(batch_0.pos,batch_0.batch,size=voxel_size,start=start,end=end)
    voxel_cluster_0, perm_0 = consecutive_cluster(batch_0_voxels)
    batch_sample_0 = batch_0.batch[perm_0]
    batch_1_voxels = voxel_grid(batch_1.pos,batch_1.batch,size=voxel_size,start=start,end=end)
    voxel_cluster_1, perm_1 = consecutive_cluster(batch_1_voxels)
    batch_sample_1 = batch_1.batch[perm_1]
    
    return batch_0,batch_0_voxels,batch_sample_0,batch_1,batch_1_voxels,batch_sample_1


if __name__ == '__main__':
    for x in range(100):
        batch = [(torch.rand(100000,6),torch.rand(100000,6)) for x in range(20)]


        result = collate_voxel(batch=batch,voxel_size=1/2,input_dim=6,start=None,end=None)



    points = load_las(r'/mnt/cm-nas03/synch/students/sam/data_test/2018/0_0_5D696L9N.las')
    circles = circle_split(points,2,center = False,clearance = 20)
    circles = [x for x in circles if x.shape[0]>100]
    circle_list =[]
    rand_colors = [np.random.uniform(0,1,(3))+np.zeros_like(x[:,:3]) for x in circles]
    for index,circle in enumerate(circles):
        rgb = rand_colors[index]
        circle[:,3:] = rgb
        circle_list.append(circle)
    
    all_circles = np.concatenate(circle_list,axis=0)
    save_las(all_circles[:,:3],'/mnt/cm-nas03/synch/students/sam/test_circles.las',all_circles[:,3:])

