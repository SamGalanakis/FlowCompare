"""
=================================
K-NN classification - PyTorch API
=================================

The :mod:`.argKmin(K)` reduction supported by KeOps :class:`pykeops.torch.LazyTensor` allows us
to perform **bruteforce k-nearest neighbors search** with four lines of code.
It can thus be used to implement a **large-scale** 
`K-NN classifier <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_,
**without memory overflows**.



"""

#####################################################################
# Setup
# -----------------
# Standard imports:

import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from pykeops.torch import LazyTensor
import pykeops
from models import DGCNNembedder 
#pykeops.clean_pykeops()          # just in case old build files are still present
#pykeops.test_torch_bindings()






use_cuda = True
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


context = torch.randn((100*2000,3)).type(dtype).contiguous()

x = torch.randn((12*2000,3)).type(dtype).contiguous()


######################################################################
# K-Nearest Neighbors search
# ----------------------------

#####################################################################
# Peform the K-NN classification, with a fancy display:
#
start = time.time()
X_i = LazyTensor(x[:, None, :])  # (10000, 1, 784) test set
X_j = LazyTensor(context[None, :, :])  # (1, 60000, 784) train set


D_ij = ((X_i - X_j) ** 2).sum(
    -1
)
ind_knn = D_ij.argKmin(100, dim=1)
if use_cuda:
    torch.cuda.synchronize()
end = time.time()
print("{:.2f}s.".format(end - start))
pass

