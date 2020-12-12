import torch
from vedo import show, Points,settings
import numpy as np
import time

#Losses from original repo
def loss_fun(z, z_ldetJ, prior_z, e, e_ldetJ, prior_e):
    ll_z = prior_z.log_prob(z.cpu()).to(z.device) + z_ldetJ
    ll_e = prior_e.log_prob(e.cpu()).to(e.device) + e_ldetJ
    return -torch.mean(ll_z), -torch.mean(ll_e)


def loss_fun_ret(z, z_ldetJ, prior_z):
    ll_z = prior_z.log_prob(z.cpu()).to(z.device) + z_ldetJ
    return -torch.mean(ll_z)






def view_cloud(points,rgb_array=False):
    """Colorize a large cloud of 1M points by passing
    colors and transparencies in the format (R,G,B,A)
    """


    settings.renderPointsAsSpheres = False
    settings.pointSmoothing = False
    settings.xtitle = 'red axis'
    settings.ytitle = 'green axis'
    settings.ztitle = 'blue*alpha axis'

    points = points.reshape(-1,3)

    if not rgb_array:
        if isinstance(points,torch.Tensor):
            points= points.cpu()
        RGB = np.zeros_like(points) + 0
    Alpha = np.ones_like(RGB[:,0])*255
    RGBA = np.c_[RGB, Alpha]  # concatenate

    
    

    # passing c in format (R,G,B,A) is ~50x faster
    points = Points(points, r=2, c=RGBA) #fast
    #pts = Points(pts, r=2, c=pts, alpha=pts[:, 2]) #slow


    show(points, __doc__, axes=True)