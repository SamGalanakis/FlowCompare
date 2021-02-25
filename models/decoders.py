import torch
from torch import nn

class FullyConnectedGenerator(nn.Module):
    def __init__(self, latent_dim,n_points_out,out_dim,use_bias=True):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.n_points_out = n_points_out
        self.use_bias = use_bias

        self.layers = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=64,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=self.out_dim * self.n_points_out,
                      bias=self.use_bias),
        )

    def forward(self, z):
        output = self.layers(z.squeeze())
        output = output.view(-1, self.out_dim, self.n_points_out)
        return output


if __name__ == '__main__':
    decoder = FullyConnectedGenerator(32,2000,6,True)
    batch = torch.randn((100,32))
    out = decoder(batch)
    print()


