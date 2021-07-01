from numpy.core.numeric import Inf
import torch
import numpy as np
import time
from laspy.file import File
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import math
import laspy
import torch
import torch.nn.functional as F
import yaml
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import open3d as o3d
import einops



# Losses from original repo

eps = 1e-8


def load_las(path, extra_dim_list=None, scale_colors=True):
    """Load las/laz from given path, laz fiels require laszip on path"""
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
        color_div = 65536
    else:
        color_div = 1
    p_X = np.array((point_records['point']['X'] * las_scaleX) + las_offsetX)
    p_Y = np.array((point_records['point']['Y'] * las_scaleY) + las_offsetY)
    p_Z = np.array((point_records['point']['Z'] * las_scaleZ) + las_offsetZ)
    try:
        points = np.vstack((p_X, p_Y, p_Z, input_las.red/color_div,
                           input_las.green/color_div, input_las.blue/color_div)).T
    except:
        pass

    return points


def bits_per_dim(log_likelihood, dims_prod):
    multiplier = torch.log(torch.Tensor([2])).to(log_likelihood.device)
    bpd = -log_likelihood * multiplier / dims_prod
    return bpd


def figure_dash(fig):
    app = dash.Dash(name='plot_fig', suppress_callback_exceptions=False)
    app.layout = html.Div([dcc.Graph(id='fig', figure=fig, style={
                          'width': '100vw', 'height': '100vh'})], style={'width': '100vw', 'height': '100vh'})
    app.run_server(debug=True)


def view_cloud_plotly(points, rgb=None, fig=None, point_size=5, show=True, axes=False, show_scale=False, colorscale=None, title=None):
    """Creat plotly figure of given cloud"""
    if isinstance(points, torch.Tensor):
        points = points.cpu()
        points = points.detach().numpy()
    if isinstance(rgb, torch.Tensor):

        rgb = rgb.cpu().detach().numpy()
    if rgb is None:
        rgb = np.zeros_like(points)
    else:
        divide_by = np.maximum(rgb.max(axis=0), eps)
        rgb = np.rint(np.divide(rgb, divide_by)*255).astype(np.uint8)
    if fig == None:
        fig = go.Figure()
    if points.shape[1] == 2:
        z = np.zeros_like(points[:, 0])
    else:
        z = points[:, 2]
    fig.add_scatter3d(
        x=points[:, 0],
        y=points[:, 1],
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
                xaxis=dict(showticklabels=False, visible=False),
                yaxis=dict(showticklabels=False, visible=False),
                zaxis=dict(showticklabels=False, visible=False)
            ))

    if title != None:
        fig.update_layout(title_text=title)

    if show:
        figure_dash(fig)
        # fig.show()
    return fig



def extract_area(full_cloud, center, clearance, shape='circle'):
    """Extract area from cloud at given center with distance clearance of shape circle or square"""
    if isinstance(full_cloud, np.ndarray):
        full_cloud = torch.from_numpy(full_cloud)
    if shape == 'square':
        x_mask = ((center[0]+clearance) > full_cloud[:, 0]
                  ) & (full_cloud[:, 0] > (center[0]-clearance))
        y_mask = ((center[1]+clearance) > full_cloud[:, 1]
                  ) & (full_cloud[:, 1] > (center[1]-clearance))
        mask = x_mask & y_mask
    elif shape == 'circle':
        mask = torch.linalg.norm(full_cloud[:, :2]-center, axis=1) < clearance
    else:
        raise Exception("Invalid shape")
    return mask



def get_voxel(cloud, center, dimensions, return_mask=False):
    """Get voxel or voxel mask from cloud at given center with specified dimensions"""
    mask = (cloud[:, :3] >= (center-dimensions/2)
            ).all(dim=1) & (cloud[:, :3] <= (center+dimensions/2)).all(dim=1)
    if return_mask:
        return mask
    else:
        return cloud[mask]



def random_subsample(points, n_samples):
    """Uniform sample of n_samples points from given cloud"""
    if points.shape[0] == 0:
        print('No points found for random sampling, replacing with dummy')
        points = np.zeros((1, points.shape[1]))
    # No point sampling if already
    if n_samples < points.shape[0]:
        random_indices = np.random.choice(
            points.shape[0], n_samples, replace=False)
        points = points[random_indices, :]

    return points




class Early_stop:
    def __init__(self, patience=50, min_perc_improvement=0):
        self.patience = patience
        self.min_perc_improvement = min_perc_improvement
        self.not_improved = 0
        self.best_loss = torch.tensor(1e+8)
        self.last_loss = self.best_loss
        self.step = -1
        self.loss_hist = []

    def log(self, loss):
        self.loss_hist.append(loss)
        self.step += 1
        # Check if improvement by sufficient margin
        if (torch.abs(self.last_loss-loss) > torch.abs(self.min_perc_improvement*self.last_loss)) & (self.last_loss < loss):
            self.not_improved = 0
        else:
            self.not_improved += 1

        # Keep track of best loss
        if loss < self.best_loss:
            self.best_loss = loss
        # Set last_loss
        self.last_loss = loss
        if self.not_improved > self.patience:
            stop_training = True
        else:
            stop_training = False
        return stop_training


def save_las(pos, path, rgb=None, extra_feature=None, feature_name='Change'):
    """Save a las file with coordinates pos and rgb colors, optional extra features"""
    hdr = laspy.header.Header(point_format=2)

    outfile = laspy.file.File(path, mode="w", header=hdr)
    if not isinstance(extra_feature, type(None)):
        outfile.define_new_dimension(
            name=feature_name,
            data_type=10,
            description="Change metric"
        )

        outfile.writer.set_dimension('change', extra_feature)

    allx = pos[:, 0]  # Four Points
    ally = pos[:, 1]
    allz = pos[:, 2]

    xmin = np.floor(np.min(allx))
    ymin = np.floor(np.min(ally))
    zmin = np.floor(np.min(allz))

    outfile.header.offset = [xmin, ymin, zmin]
    outfile.header.scale = [0.001, 0.001, 0.001]

    outfile.x = allx
    outfile.y = ally
    outfile.z = allz

    if not isinstance(rgb, type(None)):
        print('Adding color')
        if rgb.max() <= 1.0:
            rgb *= 255
        if rgb.shape[-1] == 3:
            outfile.red = rgb[:, 0]
            outfile.green = rgb[:, 1]
            outfile.blue = rgb[:, 2]
        else:
            rgb = None

    outfile.close()


def co_min_max(tensor_list):
    """Joint min max normalization"""
    is_numpy = isinstance(tensor_list[0], np.ndarray)
    if is_numpy:
        tensor_list = [torch.from_numpy(x) for x in tensor_list]
    overall_max = torch.max(torch.stack(
        [x[:, :3].max(axis=0)[0] for x in tensor_list]), dim=0)[0]
    overall_min = torch.min(torch.stack(
        [x[:, :3].min(axis=0)[0] for x in tensor_list]), dim=0)[0]
    denominator = overall_max-overall_min + eps
    for x in tensor_list:
        x[:, :3] = (x[:, :3] - overall_min)/denominator
        is_valid(x)

    if is_numpy:
        tensor_list = [x.numpy() for x in tensor_list]
    return tensor_list


def unit_sphere(points, return_inverse=False):
    """Normalize cloud to zero mean and within unit ball"""
    mean = points[:, :3].mean(axis=0)
    points[:, :3] -= mean
    furthest_distance = torch.max(torch.linalg.norm(points[:, :3], dim=-1))
    points[:, :3] = points[:, :3] / furthest_distance
    if return_inverse:
        inverse = {'furthest_distance':furthest_distance,'mean':mean}
        return points, inverse
    else:
        return points

def co_unit_sphere(points_0, points_1, return_inverse=False):
    """Joint zero mean unit ball normalization"""
    l_0 = points_0.shape[0]

    joint, inverse = unit_sphere(
        torch.cat((points_0, points_1)), return_inverse=True)
    if return_inverse:
        return joint[:l_0, :], joint[l_0:], inverse
    else:
        return joint[:l_0, :], joint[l_0:]



def view_cloud_o3d(xyz, rgb, show=True):
    """Visualize cloud with o3d"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    if show:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def exp_from_paper(x, eps):
    """
    Calculate matrix expoential as done in : http://proceedings.mlr.press/v119/xiao20a.html
    compute the matrix exponential: \sum_{k=0}^{\infty}\frac{x^{k}}{k!}
    """

    scale = int(
        np.ceil(np.log2(np.max([torch.norm(x, p=1, dim=-1).max().item(), 0.5]))) + 1)
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


def rotate_xy(rad):
    """Create rotation matrix of given rad in 2 dim"""
    matrix = torch.tensor(
        [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])


def expm(x, eps, algo='torch'):
    if algo == 'torch':
        return torch.matrix_exp(x)
    elif algo == 'original':
        return exp_from_paper(x, eps)
    else:
        raise Exception('Invalid expm algo!')



def rgb_to_hsv(rgb, scale_after=False):
    """Convert rgb values to hsv with optional scaling"""
    hsv = torch.zeros_like(rgb)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    cmax = rgb.max(axis=1)[0]
    cmin = rgb.min(axis=1)[0]
    v = cmax
    hsv[cmax == cmin, 2] = v[cmax == cmin]
    s = (cmax-cmin) / (cmax + eps)

    rc = (cmax-r) / (cmax-cmin + eps)
    gc = (cmax-g) / (cmax-cmin + eps)
    bc = (cmax-b) / (cmax-cmin + eps)
    mask_0 = (r == cmax)
    mask_1 = (g == cmax)
    mask_neither = torch.logical_not(torch.logical_or(mask_0, mask_1))
    hsv[mask_0, 0] = (bc-gc)[mask_0]
    hsv[mask_1, 0] = (2.0+rc-bc)[mask_1]
    hsv[mask_neither, 0] = (4.0+gc-rc)[mask_neither]
    hsv[:, 0] = (hsv[:, 0]/6.0) % 1.0
    hsv[:, 1] = s
    hsv[:, 2] = cmax
    if scale_after:
        hsv = hsv * torch.Tensor([360, 100, 100])
    return hsv




def oversample_cloud(cloud, n_points):
    """Oversample cloud if too few points by repeating the number of missing points to reach the target randomly"""
    n_points_original = cloud.shape[0]
    if n_points_original >= n_points:
        return cloud
    else:
        random_indices = torch.randint(
            0, n_points_original, (n_points - n_points_original,), device=cloud.device)
        return torch.cat((cloud, cloud[random_indices, ...]))


def config_loader(path):
    """Load yaml config"""
    with open(path) as f:
        raw_dict = yaml.load(f,Loader=yaml.FullLoader)
    return {key: raw_dict[key]['value'] for key in raw_dict.keys()}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_prob_to_color(log_prob_1_given_0, log_prob_0_given_0, multiple=3.):
    base_mean = log_prob_0_given_0.mean()
    base_std = log_prob_0_given_0.std()
    print(f'Base  mean: {base_mean.item()}, base_std: {{base_std.item()}}')
    changed_mask_1 = torch.abs(
        log_prob_1_given_0-base_mean) > multiple*base_std
    log_prob_1_given_0 += torch.abs(log_prob_1_given_0.min())
    log_prob_1_given_0[~changed_mask_1] = 0
    return log_prob_1_given_0




def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def mean_except_batch(x, num_dims=1):
    '''
    Averages all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_mean: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).mean(-1)




class Scheduler:
    def __init__(self, optimizer, mem_iter, factor, threshold, min_lr, verbose=True):
        self.mem_iter = mem_iter
        self.factor = factor
        self.threshold = threshold
        self.current_average = 0
        self.prev_average = Inf  # Don't go down in lr on first check
        self.step_counter = 0
        self.optimizer = optimizer
        self.verbose = verbose
        self.min_lr = min_lr

    def set_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = max(g['lr'] * self.factor, self.min_lr)

    def threshold_val(self):
        return self.prev_average - abs(self.prev_average)*(self.threshold)

    def step(self, loss):
        loss = loss.item()
        self.step_counter += 1

        self.current_average = self.current_average * \
            (self.step_counter-1)/self.step_counter + loss / self.step_counter

        if self.step_counter >= self.mem_iter:
            self.step_counter = 0
            if self.threshold_val() <= self.current_average:  # If within threshold or higher loss than b4, change lr
                self.set_lr()
                if self.verbose:
                    print(
                        f"Updating lr as prev: {self.prev_average} new: {self.current_average} ")
            self.prev_average = self.current_average


def rotate_xy(rad):
    matrix = torch.tensor(
        [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
    return matrix


def is_valid(tensor):
    assert not torch.logical_or(
        tensor.isnan(), tensor.isinf()).any(), 'Invalid values!'

def get_voxel_index(point,min,max,sizes):
    axis_size = torch.ceil((max-min) / sizes)
    n_total_per_axis = torch.Tensor([axis_size[0:index].prod() for index in range(len(axis_size))])
    n_per_axis = ((point - min)//sizes).long()
    index =  (n_total_per_axis*n_per_axis).sum()
    return index

def get_voxel_center(point,min,sizes):
    """Get voxel center given start of voxel,a point from the voxel and the voxel sizes"""
    n_per_axis = ((point - min)//sizes)
    min_side = min + (n_per_axis * sizes)
    center  = min_side + sizes/2
    return center

def get_all_voxel_centers(start,end,size):
    """Get all voxel centers given start end (min-max) and voxel sizes"""
    n_dims = len(size)
    axis_centers = [torch.arange(start[i] + size[i] / 2, end[i] + size[i] / 2, size[i]) for i in range(n_dims)]
    centers = torch.stack(torch.meshgrid(*axis_centers[::-1])).reshape((n_dims,-1)).T.flip(-1)
    return centers 



