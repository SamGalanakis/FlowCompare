import torch
import numpy as np
from dataloaders import AmsGridLoader
import laspy

#inFile = laspy.file.File('/media/raid/sam/ams_dataset/5D74DXOT.laz', mode='r')


dataset = AmsGridLoader('/media/raid/sam/ams_dataset/','/media/raid/sam/processed_ams',grid_square_size=4,clearance=10,preload=True,min_points=2000,subsample='fps',height_min_dif=0.5,normalization='co_unit_sphere',grid_type='circle')
pass