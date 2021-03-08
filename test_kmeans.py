import torch
from utils import *
import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor


def KMeans(x, K=10, Niter=10, verbose=True):
    use_cuda=torch.cuda.is_available()
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_path =r"save\conditional_flow_compare\young-leaf-836_0_6864_model_dict.pt"
    points_0 = load_las(r"D:\data\cycloData\2016\0_5D4KVPBP.las")[:,:3]
    points_1 = load_las(r"D:\data\cycloData\2020\0_WE1NZ71I.las")[:,:3]
    sign_point = np.array([86967.46,439138.8])
    sign_0 = extract_area(points_0,sign_point,4,'circle')
    sign_0 = torch.from_numpy(sign_0.astype(dtype=np.float32)).to(device)
    sign_0 = random_subsample(sign_0,3000)
    cl,c = KMeans(sign_0,10)
    print()