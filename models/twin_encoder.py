from gcn_encoder import GCNEncoder
import torch
import torch.nn as nn
from utils import MLP




class TwinEncoder(nn.Module):
    def __init__(self,input_dim,n_classes,encoder,intermediate_latent=1024):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.encoder = encoder
        self.intermediate_latent = intermediate_latent
        self.mlp_after_combine = nn.Sequential(
            MLP([self.intermediate_latent*2, 512]), nn.Dropout(0.5), MLP([512, 256]), nn.Dropout(0.5),
            nn.Linear(256, self.n_classes))



    def forward(self,data_0,data_1):
        encoding_0 = self.encoder(data_0)
        encoding_1 = self.encoder(data_1)

        combined_encodings = torch.cat((encoding_0,encoding_1),dim=-1)
        
        return self.mlp_after_combine(combined_encodings)

if __name__ == '__main__':
    encoder = GCNEncoder(6,32)
    encoder = TwinEncoder(6,5,encoder,32)
    