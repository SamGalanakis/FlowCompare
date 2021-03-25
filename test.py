import torch
import numpy as np
eps=1e-8


def expm(x):
    """
    compute the matrix exponential: \sum_{k=0}^{\infty}\frac{x^{k}}{k!}
    """
    scale = int(np.ceil(np.log2(np.max([torch.norm(x, p=1, dim=-1).max().item(), 0.5]))) + 1)
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


x = torch.randn((100,100))

y = expm(x)
pass