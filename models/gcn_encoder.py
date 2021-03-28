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
    return torch.nn.Sequential(*[
        torch.nn.Sequential(torch.nn.Linear(channels[i - 1], channels[i]), torch.nn.ReLU(), torch.nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class GCNembedder(torch.nn.Module):
    def __init__(self,in_dim, out_channels, k=20, aggr='max'):
            super().__init__()
            self.in_dim =in_dim

            self.conv1 = DynamicEdgeConv(MLP([2 * self.in_dim, 64, 64]), k, aggr)
            self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
            self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)

            
            self.mlp = Seq(
               MLP([3 * 64, 1024],batch_norm=False),  MLP([1024, 512],batch_norm=False), MLP([512, 256],batch_norm=False),MLP([256, out_channels],batch_norm=False))

    def forward(self,points):
        batch_size, n_points , _ = points.shape
        batch = torch.cat([torch.ones(points.shape[1])*i for i in range(0,points.shape[0])]).long().to(points.device)
        
        points = points.reshape(-1,points.shape[-1])
        x,pos =  points[...,3:],points[...,:3]
        #Check if there are features before concat
        if x is not None:
            x0 = torch.cat([x, pos], dim=-1)
        else:
            x0 = pos
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return out.reshape((batch_size,n_points,-1))
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

    def forward(self, x,pos,batch):
       
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
    encoder = GCNembedder(6,40)
    x = torch.randn((20,10,6))
  
    
    result = encoder(x)
    pass
