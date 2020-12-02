import torch
import torch.nn as nn
from torch import distributions
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.flow_modules import (
    CouplingLayer,
    AffineCouplingFunc,
    ConditionalNet,
    StraightNet,
)
from models.point_encoders import PointnetEncoder
from utils import loss_fun , loss_fun_ret

from data.datasets_pointflow import (
    CIFDatasetDecorator,
    ShapeNet15kPointClouds,
    CIFDatasetDecoratorMultiObject,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prior_z = distributions.MultivariateNormal(
    torch.zeros(3), torch.eye(3)
)
emb_dim = 32
prior_e = distributions.MultivariateNormal(
    torch.zeros(emb_dim), torch.eye(emb_dim)
)

cloud_pointflow = ShapeNet15kPointClouds(
    tr_sample_size=2048,
    te_sample_size=2048,
    root_dir='data\\ShapeNetCore.v2.PC15k',
  
    normalize_per_shape=False,
    normalize_std_per_axis=False,
    split="train",
    scale=1.0,
    categories=["airplane"],
    random_subsample=True,
)


print('done')
