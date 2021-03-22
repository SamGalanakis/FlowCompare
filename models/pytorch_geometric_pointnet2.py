import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn,)

    def forward(self, x, pos, batch,centers = None,centers_batch=None):
        if centers==None:
            idx = fps(pos, batch, ratio=self.ratio)
            centers = pos[idx]
            centers_batch = batch[idx]
        
        
        row, col = radius(pos, centers, self.r, batch, centers_batch,
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, centers), edge_index)
        pos, batch = centers, centers_batch
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Pointnet2(torch.nn.Module):
    def __init__(self,feature_dim,out_dim=32):
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        super(Pointnet2, self).__init__()

        
        self.sa1_module = SAModule(0.5, 0.2, MLP([3+self.feature_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, self.out_dim)

    def forward(self, data):
        x,pos,batch = data.x,data.pos,data.batch
        sa1_out = self.sa1_module(x,pos,batch)

        sa2_out = self.sa2_module(*sa1_out)

        sa3_out = self.sa3_module(*sa2_out)

        x, pos, batch = sa3_out
        #print(x.min(),x.max())
        x = F.relu(self.lin1(x))
        assert not x.isnan().any()
        x = F.dropout(x, p=0.5, training=self.training)
        assert not x.isnan().any()
        x = F.relu(self.lin2(x))
        assert not x.isnan().any()
        x = F.dropout(x, p=0.5, training=self.training)
        assert not x.isnan().any()
        x = self.lin3(x)
        assert not x.isnan().any()
        return x


    
def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data.x,data.pos,data.batch), data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Pointnet2(feature_dim=0).to(device)
    class inputs:
        def __init__(self):
            pass
    x = inputs()
    inputs = x.pos = torch.randn((100,3)).to(device)
    x.x = None
    x.batch = torch.zeros(100).to(device).long()
    y = model(x)