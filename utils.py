import torch
from vedo import show, Points,settings
import numpy as np
import time
import pylas 
from laspy.file import File
from pyntcloud import PyntCloud
import pandas as pd

from matplotlib import widgets
from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib.artist as artist
import matplotlib.image as image
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


def compare_clouds(extraction_1,extraction_2,class_labels):
    rgb1 =     np.rint(np.divide(extraction_1[:,3:],extraction_1[:,3:].max(axis=0))*255).astype(np.uint8)
    rgb2 =     np.rint(np.divide(extraction_2[:,3:],extraction_2[:,3:].max(axis=0))*255).astype(np.uint8)
    
    # class_to_return = None
    points1 = extraction_1[:,:3]
    
    points2 = extraction_2[:,:3]
    points2[:,0]+=10
    axes = plt.axes(projection='3d')
    axes.scatter(points1[:,0], points1[:,1], points1[:,2], c = rgb1/255, s=0.1)

    axes.scatter(points2[:,0], points2[:,1], points2[:,2], c = rgb2/255, s=0.1)
    plt.axis('off')
    
    props = ItemProperties(labelcolor='black', bgcolor='yellow',
                        fontsize=10, alpha=0.2)
    hoverprops = ItemProperties(labelcolor='white', bgcolor='blue',
                                fontsize=10, alpha=0.2)

    menuitems = []

    def on_select(item):
        global class_to_return
        class_to_return = item.labelstr
        plt.close()
        print('you selected %s' % item.labelstr)
    for label in class_labels:
      
        item = MenuItem(plt.gcf(), label, props=props, hoverprops=hoverprops,
                        on_select=on_select)
        menuitems.append(item)

    menu = Menu(plt.gcf(), menuitems)
    plt.show()
    

    return class_to_return


    
    

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
    random_indices = np.random.choice(points.shape[0],n_samples, replace=True)
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


class ItemProperties:
    def __init__(self, fontsize=14, labelcolor='black', bgcolor='yellow',
                 alpha=1.0):
        self.fontsize = fontsize
        self.labelcolor = labelcolor
        self.bgcolor = bgcolor
        self.alpha = alpha

        self.labelcolor_rgb = colors.to_rgba(labelcolor)[:3]
        self.bgcolor_rgb = colors.to_rgba(bgcolor)[:3]


class MenuItem(artist.Artist):
    parser = mathtext.MathTextParser("Bitmap")
    padx = 5
    pady = 5

    def __init__(self, fig, labelstr, props=None, hoverprops=None,
                 on_select=None):
        artist.Artist.__init__(self)

        self.set_figure(fig)
        self.labelstr = labelstr

        if props is None:
            props = ItemProperties()

        if hoverprops is None:
            hoverprops = ItemProperties()

        self.props = props
        self.hoverprops = hoverprops

        self.on_select = on_select

        x, self.depth = self.parser.to_mask(
            labelstr, fontsize=props.fontsize, dpi=fig.dpi)

        if props.fontsize != hoverprops.fontsize:
            raise NotImplementedError(
                'support for different font sizes not implemented')

        self.labelwidth = x.shape[1]
        self.labelheight = x.shape[0]

        self.labelArray = np.zeros((x.shape[0], x.shape[1], 4))
        self.labelArray[:, :, -1] = x/255.

        self.label = image.FigureImage(fig, origin='upper')
        self.label.set_array(self.labelArray)

        # we'll update these later
        self.rect = patches.Rectangle((0, 0), 1, 1)

        self.set_hover_props(False)

        fig.canvas.mpl_connect('button_release_event', self.check_select)

    def check_select(self, event):
        over, junk = self.rect.contains(event)
        if not over:
            return

        if self.on_select is not None:
            self.on_select(self)

    def set_extent(self, x, y, w, h):
        
        self.rect.set_x(x)
        self.rect.set_y(y)
        self.rect.set_width(w)
        self.rect.set_height(h)

        self.label.ox = x + self.padx
        self.label.oy = y - self.depth + self.pady/2.

        self.hover = False

    def draw(self, renderer):
        self.rect.draw(renderer)
        self.label.draw(renderer)

    def set_hover_props(self, b):
        if b:
            props = self.hoverprops
        else:
            props = self.props

        r, g, b = props.labelcolor_rgb
        self.labelArray[:, :, 0] = r
        self.labelArray[:, :, 1] = g
        self.labelArray[:, :, 2] = b
        self.label.set_array(self.labelArray)
        self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

    def set_hover(self, event):
        """
        Update the hover status of event and return whether it was changed.
        """
        b, junk = self.rect.contains(event)

        changed = (b != self.hover)

        if changed:
            self.set_hover_props(b)

        self.hover = b
        return changed


class Menu:
    def __init__(self, fig, menuitems):
        self.figure = fig
        fig.suppressComposite = True

        self.menuitems = menuitems
        self.numitems = len(menuitems)

        maxw = max(item.labelwidth for item in menuitems)
        maxh = max(item.labelheight for item in menuitems)

        x0 = 100
        y0 = 400

        width = maxw + 2*MenuItem.padx
        height = maxh + MenuItem.pady

        for item in menuitems:
            left = x0
            bottom = y0 - maxh - MenuItem.pady

            item.set_extent(left, bottom, width, height)

            fig.artists.append(item)
            y0 -= maxh + MenuItem.pady

        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        draw = False
        for item in self.menuitems:
            draw = item.set_hover(event)
            if draw:
                self.figure.canvas.draw()
                break



if __name__ == "__main__":
    points = load_las("D:/data/cycloData/2016/0_5D4KVPBP.las")
    #grid_split(points,5000)
    #view_cloud(points[:,:3],points[:,3:],subsample = False)
    #view_cloud(points[:,:3],points[:,3:],subsample = False)
    #view_cloud(points[:,:3],points[:,3:],subsample = False)
    
    sign_point = np.array([86967.46,439138.8])
    points = extract_area(points,sign_point,1.5,'cylinder')
    points = random_subsample(points,3000)
    class_return = compare_clouds(points[:,0:3],points[:,3:],['change','nochange'])
    print(class_return)
    view_cloud(points[:,:3],points[:,3:],subsample = False)
