import torch
from torch import nn


class VoxelCNN(nn.Module):
    def __init__(self,input_dim,emb_dim,p_drop=0):
        super().__init__()

        self.layers = nn.Sequential(
        nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.Dropout3d(p_drop),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.MaxPool3d(2),  # 16
            nn.Conv3d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ELU(),
            nn.Conv3d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ELU(),
            nn.MaxPool3d(2),  # 8
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ELU(),
            nn.Conv3d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ELU(),
            nn.MaxPool3d(2),  # 4
            nn.Conv3d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            nn.Conv3d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            nn.MaxPool3d(2),  # 2
            nn.Conv3d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            nn.Conv3d(512, 1024, 2, padding=0, bias=False),
            nn.BatchNorm3d(1024),
            nn.ELU(),
        )

    def forward(self,x):
        return self.layers(x).squeeze()

if __name__ == '__main__':
    encoder = VoxelCNN(6,32)
    x = torch.randn((100,32,32,32,32))
    result = encoder(x)
    print(result.shape)

        