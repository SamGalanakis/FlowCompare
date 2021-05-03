from numpy.core.numeric import Inf
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
import open3d
from yaml import load as load_yaml
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
    #Will work with laz if laszip on path 
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
def bits_per_dim(log_likelihood,dims_prod):
    multiplier = torch.log(torch.Tensor([2])).to(log_likelihood.device)
    bpd = -log_likelihood * multiplier / dims_prod
    return bpd

def remove_outliers(array,n_neighbors=10,std_ratio=2.0):
    array=array.numpy()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(array[:,:3])
    pcd.remove_statistical_outlier(nb_neighbors=n_neighbors,std_ratio=std_ratio)
    array[:,:3] = np.asarray(pcd.points)
    return torch.from_numpy(array)
def view_cloud_plotly(points,rgb=None,fig=None,point_size=8,show=True,axes=False,show_scale=False,colorscale=None,title=None):
    if  isinstance(points,torch.Tensor):
        points = points.cpu()
        points = points.detach().numpy()
    if  isinstance(rgb,torch.Tensor):
        
        rgb = rgb.cpu().detach().numpy()
    if  rgb is None:
        rgb = np.zeros_like(points)
    else:
        divide_by = np.maximum(rgb.max(axis=0),eps)
        rgb = np.rint(np.divide(rgb,divide_by)*255).astype(np.uint8)
    if fig==None:
        fig = go.Figure()
    if points.shape[1]==2:
        z=np.zeros_like(points[:,0])
    else:
        z=points[:,2]
    fig.add_scatter3d(
        x=points[:,0], 
        y=points[:,1], 
        z=z, 
        marker=dict(
        size=point_size,
        color=rgb,  
        colorscale=colorscale,
        showscale=show_scale,
        opacity=1,
    ), 
        
        opacity=1, 
        mode='markers',
        
    )


    if not axes:
        fig.update_layout(
        scene=dict(
        xaxis=dict(showticklabels=False,visible= False),
        yaxis=dict(showticklabels=False,visible= False),
        zaxis=dict(showticklabels=False, visible= False)
        ))

    if title!=None:
        fig.update_layout(title_text=title)

    if show:
        fig.show()
    return fig
    
  
def bin_probs(log_probs_0,log_probs_1,n_bins=50):
    
    std0 = log_probs_0.std()
    m0 = log_probs_0.mean()
    bin_width = 0.5*std0
    bin_counts = [0]*n_bins
    bins,counts = torch.unique(torch.abs(log_probs_1-m0)//bin_width,return_counts=True)
    normalized_counts = counts/log_probs_1.shape[0]
    for bin,normalized_count in list(zip(bins,normalized_counts)):
        if int(bin)>=n_bins:
            break
        bin_counts[int(bin)] = normalized_count.item()
        

    

    return bin_counts
    

def extract_area(full_cloud,center,clearance,shape= 'circle'):
    if isinstance(full_cloud,np.ndarray):
        full_cloud = torch.from_numpy(full_cloud)
    if shape == 'square':
        x_mask = ((center[0]+clearance)>full_cloud[:,0]) &   (full_cloud[:,0] >(center[0]-clearance))
        y_mask = ((center[1]+clearance)>full_cloud[:,1]) &   (full_cloud[:,1] >(center[1]-clearance))
        mask = x_mask & y_mask
    elif shape == 'circle':
        mask = torch.linalg.norm(full_cloud[:,:2]-center,axis=1) <  clearance
    else:
        raise Exception("Invalid shape")
    return mask

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
    x = torch.arange(center_x-clearance, center_x+clearance, circle_radius*2,device = points.device) 
    y = torch.arange(center_y-clearance, center_y+clearance, circle_radius*2,device = points.device) 
   
    circles_list = []
    for x_val in x:
        for y_val in y:
            tile = points[torch.linalg.norm(points[:,:2]-torch.Tensor([x_val,y_val]).to(points.device),dim=1,ord=2)<circle_radius]
            circles_list.append(tile)
    return circles_list


def random_subsample(points,n_samples):
    if points.shape[0]==0:
        print('No points found for random sampling, replacing with dummy')
        points = np.zeros((1,points.shape[1]))
    #No point sampling if already 
    if n_samples < points.shape[0]:
        random_indices = np.random.choice(points.shape[0],n_samples, replace=False)
        points = points[random_indices,:]
    
    return points



def ground_remover(points,height_bin=0.3,max_below =1):
    lengths =np.arange(points[:,2].min(),points[:,2].max(),height_bin).tolist()
    n_points = []

    for x in lengths:
        n_points.append(sum((points[:,2]>x) & (points[:,2]<x+height_bin)).item())
    if len(n_points)==0:
        height = 1e+10
        print('Only one bin, removing all!')
    zipped = sorted(zip(n_points,lengths),key=lambda x :x[0],reverse=True)
    for n_point,height in zipped:
        how_many_below = n_points.index(n_point)
        if how_many_below<=max_below:
            break
    cut_off_height = height + height_bin
    mask = points[:,2]>cut_off_height
    return mask

    


    
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
def unit_sphere(points):
    points[:,:3] -= points[:,:3].mean(axis=0)
    furthest_distance = torch.max(torch.sqrt(torch.sum(torch.abs(points[:,:3])**2,axis=-1)))
    points[:,:3] = points[:,:3] / furthest_distance
    return points

def co_unit_sphere(points_0,points_1):
    l_0 = points_0.shape[0]

    joint = unit_sphere(torch.cat((points_0,points_1)))
    return joint[:l_0,:] , joint[l_0:]
def co_standardize(tensor_0,tensor_1):
    concatenated = torch.cat((tensor_0,tensor_1),dim=0)
    concatenated = (concatenated-concatenated.mean(axis=0))/(concatenated.std(axis=0)+eps)
    tensor_0, tensor_1 = torch.split(concatenated,tensor_0.shape[0],dim=0)
    return tensor_0, tensor_1
def sep_standardize(tensor_0,tensor_1):
    return (tensor_0-tensor_0.mean(axis=0))/tensor_0.std(axis=0),(tensor_1-tensor_1.mean(axis=0))/tensor_1.std(axis=0)



def exp_from_paper(x,eps):
    """
    compute the matrix exponential: \sum_{k=0}^{\infty}\frac{x^{k}}{k!}
    """
    
    scale = int(np.ceil(np.log2(np.max([torch.norm(x, p=1, dim=-1).max().item(), 0.5]))) + 1)
    x = x / (2 ** scale)
    s = torch.eye(x.size(-1), device=x.device)
    t = x
    k = 2
    while torch.norm(t, p=1, dim=-1).max().item() > eps:
        s = s + t
        t = torch.matmul(x, t) / k
        k = k + 1
    for i in range(scale):
        s = torch.matmul(s, s) 
    return s




def expm(x,eps,algo='torch'):
    if algo=='torch':
        return torch.matrix_exp(x)
    elif algo=='original':
        return exp_from_paper(x,eps)
    else:
        raise Exception('Invalid expm algo!')



def feature_assigner(x,input_dim):
    return None if input_dim==3 else x[:,3:]

    
def MLP(channels, batch_norm=True):
    return torch.nn.Sequential(*[
        torch.nn.Sequential(torch.nn.Linear(channels[i - 1], channels[i]), torch.nn.ReLU(), torch.nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])



def rgb_to_hsv(rgb,scale_after=False):
    hsv = torch.zeros_like(rgb)
    r = rgb[:,0]
    g = rgb[:,1]
    b = rgb[:,2]
    cmax = rgb.max(axis=1)[0]
    cmin = rgb.min(axis=1)[0]
    v = cmax
    hsv[cmax==cmin,2] = v[cmax==cmin]
    s = (cmax-cmin) / (cmax +eps)

    rc = (cmax-r) / (cmax-cmin +eps)
    gc = (cmax-g) / (cmax-cmin +eps)
    bc = (cmax-b) / (cmax-cmin +eps)
    mask_0 = (r == cmax)
    mask_1 = (g == cmax)
    mask_neither = torch.logical_not(torch.logical_or(mask_0,mask_1))
    hsv[mask_0,0] = (bc-gc)[mask_0]
    hsv[mask_1,0] = (2.0+rc-bc)[mask_1]
    hsv[mask_neither,0] = (4.0+gc-rc)[mask_neither]
    hsv[:,0] = (hsv[:,0]/6.0) % 1.0
    hsv[:,1] = s
    hsv[:,2] = cmax
    if scale_after:
        hsv = hsv * torch.Tensor([360,100,100])
    return hsv


def time_labeling(extract_0,extract_1):
    n_points_0 = extract_0.shape[0]
    n_points_1 = extract_1.shape[0]
    label_points = torch.cat((torch.ones(n_points_0),-torch.ones(n_points_1))).to(extract_0.device)
    label_points = label_points.unsqueeze(-1)
    return torch.cat( (torch.cat((extract_0,extract_1),dim=0),label_points),dim=-1)
if __name__ == '__main__':
    extract_0,extract_1 = torch.randn(10,100,6),torch.randn(10,100,6)
    combined = time_labeling(extract_0,extract_1)
    test_hsv = torch.Tensor([[1,10,20],[5,30,3],[255,6,1]])
    print(test_hsv)
    test_hsv /= 255
    
    print(rgb_to_hsv(test_hsv,scale_after=True))

def oversample_cloud(cloud,n_points):
    n_points_original = cloud.shape[0]
    if n_points_original>=n_points:
        return cloud
    else:
        random_indices = torch.randint(0,n_points_original,(n_points - n_points_original,),device=cloud.device)
        return torch.cat((cloud,cloud[random_indices,...]))
def config_loader(path):
    with open(path) as f:
        raw_dict = load_yaml(f)
    return  {key:raw_dict[key]['value'] for key in raw_dict.keys()}
def rotation_z(rad):
    rot = torch.Tensor([[torch.cos(rad),-torch.sin(rad),0],[torch.sin(rad),torch.cos(rad),0],[0,0,1]])
    return rot

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

