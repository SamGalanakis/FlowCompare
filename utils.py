import torch
from vedo import show, Points,settings
import numpy as np
import time
import pylas 
from laspy.file import File


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




def extract_area(full_cloud,center,clearance,shape= 'cylinder'):
    if shape == 'square':
        x_mask = ((center[0]+clearance)>full_cloud[:,0]) &   (full_cloud[:,0] >(center[0]-clearance))
        y_mask = ((center[1]+clearance)>full_cloud[:,1]) &   (full_cloud[:,1] >(center[1]-clearance))
        mask = x_mask & y_mask
    elif shape == 'cylinder':
        mask = np.linalg.norm(full_cloud[:,:2]-center,axis=1) <  clearance
    return full_cloud[mask]

def grid_split(points,grid_size,center = False,clearance = 30000):
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
    random_indices = np.random.choice(points.shape[0],n_samples, replace=False)
    points = points[random_indices,:]
    
    return points
def view_cloud(points,rgb_array=False,subsample=False):
    """Colorize a large cloud of 1M points by passing
    colors and transparencies in the format (R,G,B,A)
    """

   

    settings.renderPointsAsSpheres = False
    settings.pointSmoothing = False
    settings.xtitle = 'x axis'
    settings.ytitle = 'y axis'
    settings.ztitle = 'z axis'

    
    
    points = points.reshape(-1,3)


    if isinstance(points,torch.Tensor):
        points= points.cpu().numpy()
    if subsample:
        random_indices = np.random.choice(points.shape[0],subsample, replace=False)
        points = points[random_indices,:]
        rgb_array = rgb_array[random_indices,:]

    if  isinstance(rgb_array,bool):    
        RGB = np.zeros_like(points) + 0
        Alpha = np.ones_like(RGB[:,0])*255
        
    else: 
        RGB =     np.rint(np.divide(rgb_array,rgb_array[:,:3].max(axis=0))*255).astype(np.uint8)
        if rgb_array.shape[0]==4:
            Alpha = rgb_array[:,-1]
        else:
            Alpha = np.ones_like(RGB[:,0])*255

    RGBA = np.c_[RGB, Alpha]  # concatenate
        

    
    

    # passing c in format (R,G,B,A) is ~50x faster
    points = Points(points, r=2, c=RGBA) #fast
    #pts = Points(pts, r=2, c=pts, alpha=pts[:, 2]) #slow


    show(points, __doc__, axes=True)
if __name__ == "__main__":
    points = load_las("D:/data/cycloData/2016/0_5D4KVPBP.las")
    #grid_split(points,5000)
    #view_cloud(points[:,:3],points[:,3:],subsample = False)
    #view_cloud(points[:,:3],points[:,3:],subsample = False)
    #view_cloud(points[:,:3],points[:,3:],subsample = False)
    sign_point = np.array([86967.46,439138.8])
    points = extract_area(points,sign_point,1.5,'cylinder')
    view_cloud(points[:,:3],points[:,3:],subsample = False)
