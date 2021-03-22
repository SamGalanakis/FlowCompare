import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from pytorch_geometric_pointnet2 import SAModule,MLP






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

    def forward(self,data_0,data_1):
        x_0,pos_0,batch_0 = data_0.x,data_0.pos,data_0.batch
        x_1,pos_1,batch_1 = data_1.x,data_1.pos,data_1.batch

        sa1_out_0 = self.sa1_module(x_0,pos_0,batch_0)
        centers_0 = sa1_out_0[1]
      
        batch_centers_0 = sa1_out_0[2]


        sa2_out_0 = self.sa2_module(*sa1_out_0)
        centers_1 = sa2_out_0[1]
        batch_centers_1 = sa2_out_0[2]
  
        
        sa1_out_1 = self.sa1_module(x_1,pos_1,batch_1,centers_0,batch_centers_0)
        
        sa2_out_1 = self.sa1_module(*sa1_out_1,centers_1,batch_centers_1)

        return sa2_out_0,sa2_out_1



if __name__ == '__main__':
    

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Pointnet2Partial(feature_dim=0).to(device)
    class inputs:
        def __init__(self):
            self.x = None
            self.pos = torch.randn((100,3)).to(device)
            self.batch = torch.zeros(100).to(device).long()

    y = model(inputs(),inputs())