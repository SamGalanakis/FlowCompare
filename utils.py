import torch
from vedo import show, Points,settings
import numpy as np
import time
import pylas 
from laspy.file import File
from pyntcloud import PyntCloud
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
#Losses from original repo
def loss_fun(z, z_ldetJ, prior_z, e, e_ldetJ, prior_e):
    ll_z = prior_z.log_prob(z.cpu()).to(z.device) + z_ldetJ
    ll_e = prior_e.log_prob(e.cpu()).to(e.device) + e_ldetJ
    return -torch.mean(ll_z), -torch.mean(ll_e)


def loss_fun_ret(z, z_ldetJ, prior_z):
    ll_z = prior_z.log_prob(z.cpu()).to(z.device) + z_ldetJ
    return -torch.mean(ll_z)



def load_las(path):
    input_las = File(path, mode='r')
    point_records = input_las.points.copy()
    las_scaleX = input_las.header.scale[0]
    las_offsetX = input_las.header.offset[0]
    las_scaleY = input_las.header.scale[1]
    las_offsetY = input_las.header.offset[1]
    las_scaleZ = input_las.header.scale[2]
    las_offsetZ = input_las.header.offset[2]

    # calculating coordinates
    p_X = np.array((point_records['point']['X'] * las_scaleX) + las_offsetX)
    p_Y = np.array((point_records['point']['Y'] * las_scaleY) + las_offsetY)
    p_Z = np.array((point_records['point']['Z'] * las_scaleZ) + las_offsetZ)

    points = np.vstack((p_X,p_Y,p_Z,input_las.red,input_las.green,input_las.blue)).T
    
    return points

def view_cloud_plotly(points,rgb=None,fig=None,point_size=2,show=True,axes=False,show_scale=False,colorscale=None):
    if not isinstance(rgb,np.ndarray):
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
    if not center:
        center = points[:,:2].mean(axis=0)
    #points = points[np.linalg.norm((points[:,:2]-center),axis=1)<clearance]
    
    points = extract_area(points,center,clearance,'square')
    x = np.arange(points[:,0].min(), points[:,0].max(), grid_size)
    y = np.arange(points[:,1].min(), points[:,1].max(), grid_size)
    grid_list = []
    for x_val in x[:-1]:
        for y_val in y[:-1]:
            mask_x = np.logical_and(points[:,0]>x_val,points[:,0]<x_val+grid_size)
            mask_y = np.logical_and(points[:,1]>y_val,points[:,1]<y_val+grid_size)
            tile = points[mask_x & mask_y]
            grid_list.append(tile)
    return grid_list

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

    

if __name__ == "__main__":
    points = load_las("D:/data/cycloData/2016/0_5D4KVPBP.las")
    grid_split(points,2)
    
