
import time
import laspy
import numpy as np
import torch
from matplotlib import pyplot as plt
from pykeops.torch import Vi, Vj
from pykeops.torch import LazyTensor
from utils import load_las, random_oversample,save_las,random_subsample,extract_area,view_cloud_plotly
from models import DGCNNembedder 
from dataloaders.ams_grid_loader_pointwise import voxel_downsample

def KNN_KeOps(K, metric="euclidean"):
    def fit(x_train):
        # Setup the K-NN estimator:
        
        start = time.time()

        # Encoding as KeOps LazyTensors:
        D = x_train.shape[1]
        X_i = Vi(0, D)  # Purely symbolic "i" variable, without any data array
        X_j = Vj(1, D)  # Purely symbolic "j" variable, without any data array

        # Symbolic distance matrix:
        if metric == "euclidean":
            D_ij = ((X_i - X_j) ** 2).sum(-1)
        # K-NN query operator:
        KNN_fun = D_ij.argKmin(K, dim=1)

        # N.B.: The "training" time here should be negligible.
        elapsed = time.time() - start

        def f(x_test):
            start = time.time()

            # Actual K-NN query:
            indices = KNN_fun(x_test, x_train)

            elapsed = time.time() - start

            indices = indices.cpu().numpy()
            return indices, elapsed
        return f, elapsed

    return fit

def KNN_torch_fun(x_train, x_train_norm, x_test, K):
    
    largest = False  # Default behaviour is to look for the smallest values

    
    x_test_norm = (x_test ** 2).sum(-1)
    diss = (
        x_test_norm.view(-1, 1)
        + x_train_norm.view(1, -1)
        - 2 * x_test @ x_train.t()  # Rely on cuBLAS for better performance!
    )

    return diss.topk(K, dim=1, largest=largest).indices


def KNN_torch(K):
    def fit(x_train):
        # Setup the K-NN estimator:
        x_train = torch.tensor(x_train)
        start = time.time()
        # The "training" time here should be negligible:
        x_train_norm = (x_train ** 2).sum(-1)
        elapsed = time.time() - start

        def f(x_test):
            
            start = time.time()

            # Actual K-NN query:
            out = KNN_torch_fun(x_train, x_train_norm, x_test, K)

            elapsed = time.time() - start
            indices = out
            return indices, elapsed

        return f, elapsed

    return fit


if __name__ == '__main__':
    use_cuda = True
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor





    device = 'cpu'
    context_cloud = torch.randn((100000,3)).to(device).contiguous()
    context = context_cloud[:,:3].contiguous()
    x = torch.randn((1000,3)).to(device).contiguous()
    k=1000


    for knn_func in [KNN_torch,KNN_KeOps]:
        print(str(knn_func))
        knn = knn_func(k)
        fitted_knn,elapsed_fit = knn(context)
        print(f'Fitting: {elapsed_fit}')
        index,elapsed_query = fitted_knn(x)
        print(f'Query: {elapsed_query}')
        elapsed_total = elapsed_fit+elapsed_query
        print(f'Total: {elapsed_total}')
        context_cloud = context_cloud.cpu()
        index = index.cpu()
























neighbourhoods = context_cloud[index.view((-1)),:].reshape((-1,k,context_cloud.shape[-1]))


rgb = torch.ones(context_cloud.shape)*0.0
rgb[index.reshape(-1)] = torch.tensor([1.,0,0])

fig = view_cloud_plotly(context_cloud,rgb,show=False)
fig.write_html('neighbourhoods.html')









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

ind_knn = D_ij.argKmin(1000, dim=1)
if use_cuda:
    torch.cuda.synchronize()
end = time.time()
print("{:.2f}s.".format(end - start))
pass

