import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class GCNEncoder(torch.nn.Module):
    def __init__(self,in_dim, out_channels, k=20, aggr='max'):
        super().__init__()
        self.in_dim =in_dim

        self.conv1 = DynamicEdgeConv(MLP([2 * self.in_dim, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])
        
        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        x,pos,batch = data.x,data.pos,data.batch
        #Check if there are features before concat
        if x is not None:
            x0 = torch.cat([x, pos], dim=-1)
        else:
            x0 = pos
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = global_max_pool(out,batch)
        out = self.mlp(out)
        return out
if __name__ == '__main__':
    encoder = GCNEncoder(32)
    x = torch.randn((20,2000,6))
    batch = torch.cat([torch.ones(x.shape[1])*i for i in range(0,x.shape[0])]).long()
    x = x.reshape(-1,x.shape[-1])
    result = encoder(x[...,3:],x[...,:3],batch)
    pass
