import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from .pytorch_geometric_pointnet2 import SAModule,MLP






class Pointnet2Partial(torch.nn.Module):
    def __init__(self,feature_dim,out_dim=32):
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        super().__init__()

        
        self.sa1_module = SAModule(0.5, 0.2, MLP([3+self.feature_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
  

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, self.out_dim)

    def forward(self, data):
        x,pos,batch = data.x,data.pos,data.batch
        sa1_out = self.sa1_module(x,pos,batch)

        sa2_out = self.sa2_module(*sa1_out)

        return sa2_out



if __name__ == '__main__':
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Pointnet2Partial(feature_dim=0).to(device)
    class inputs:
        def __init__(self):
            pass
    x = inputs()
    inputs = x.pos = torch.randn((100,3)).to(device)
    x.x = None
    x.batch = torch.zeros(100).to(device).long()
    y = model(x)