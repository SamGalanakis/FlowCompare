
import time
import laspy
import numpy as np
import torch
from matplotlib import pyplot as plt
from pykeops.torch import Vi, Vj
from pykeops.torch import LazyTensor
from utils import load_las, random_oversample,save_las,random_subsample,extract_area,view_cloud_plotly
from models import DGCNNembedder 



def knn_spreader(full_cloud,source,feature,k=1):
    knn = KNN_torch(k)
    knn_trained = knn(source[:,:3])
    knn_index = knn(full_cloud)
    indexed_feats = feature

    return

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

def get_knn(samples,context_cloud,n_neighbors,type='torch'):
        if type == 'torch':
            knn_func = KNN_torch
        elif type == 'KeOps':
            knn_func = KNN_KeOps
        else:
            raise Exception('Invalid knn func')
        knn =  knn_func(n_neighbors)
        knn,_ = knn(context_cloud)
        index,_  = knn(samples)
        
        return index
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

