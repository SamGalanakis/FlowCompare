import torch.nn as nn
import torch
from sklearn.datasets import make_swiss_roll
from utils import view_cloud_plotly
class Rotator(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = nn.Parameter(torch.randn((3,)))
        self.v2 = nn.Parameter(torch.randn((3,)))
       # self.translation = nn.Parameter(torch.zeros((3,1)))



    def forward(self,x):
        mean = x.mean(axis=0)
        x -= mean
        e1 = self.v1/torch.norm(self.v1)
        u2 = self.v2 - torch.dot(e1,self.v2) * e1
        e2 = u2/torch.norm(u2)

        R = torch.stack([e1,e2,torch.cross(e1,e2)],dim=1)


        return (torch.mm(R,x.T)).T + mean






