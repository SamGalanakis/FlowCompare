import torch
from torch import nn
from torch.nn import functional as F
from models import MLP



class DistributedToGlobal(nn.Module):
    def __init__(self,hidden_mlp_dims,emb_dim,out_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.out_mlp = MLP(emb_dim*2,hidden_mlp_dims,out_dim,nonlin=nn.GELU())

    def forward(self,context):
        context = context.permute(0,2,1)
        batch_size = context.shape[0]
        x1 = F.adaptive_max_pool1d(context, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(context, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)
        x = self.out_mlp(x)
        x = x.unsqueeze(1)
        return x






