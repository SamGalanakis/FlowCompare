import scipy
from scipy.linalg import logm
from scipy.stats import ortho_group, special_ortho_group

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.utils import expm, series

eps = 1e-8









class CouplingLayer(nn.Module):
    """
    Coupling layers have three types.
    Additive coupling layers: y2 = x2 + b(x1)
    Affine coupling layers: y2 = s(x1) * x2 + b(x1)
    Matrix exp coupling layers: y2 = e^{s(x1)}x2 + b(x1)
    """

    def __init__(self, flow_type, num_blocks, in_channels, hidden_channels):
        super(CouplingLayer, self).__init__()
        self.flow_type = flow_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.x2_channels = in_channels // 2
        self.x1_channels = in_channels - self.x2_channels
        if flow_type == 'additive':
            self.num_out = 1
        elif flow_type == 'affine':
            self.scale = nn.Parameter(torch.ones(1) / 8)
            self.shift = nn.Parameter(torch.zeros(1))
            self.rescale = nn.Parameter(torch.ones(1))
            self.reshift = nn.Parameter(torch.zeros(1))
            self.num_out = 2
        elif flow_type == 'matrixexp':
            self.scale = nn.Parameter(torch.ones(1) / 8)
            self.shift = nn.Parameter(torch.zeros(1))
            self.rescale = nn.Parameter(torch.ones(1) / self.x2_channels)
            self.reshift = nn.Parameter(torch.zeros(1))
            self.max_out = 24
            if self.x2_channels <= self.max_out:
                self.num_out = (self.x2_channels + 1)
            else:
                self.k = 3
                self.num_out = 2 * self.k + 1
        else:
            raise ValueError('wrong flow type')
        self.net = ConvBlock(num_blocks, self.x1_channels, self.hidden_channels, self.x2_channels * self.num_out)

    def forward(self, x, reverse=False, init=False):
        x1 = x[:, :self.x1_channels]
        x2 = x[:, self.x1_channels:]
        if self.flow_type == 'additive':
            if not reverse:
                x2 = x2 + self.net(x1, init=init)
                out = torch.cat([x1, x2], dim=1)
                log_det = x.new_zeros(x.size(0))
            else:
                x2 = x2 - self.net(x1)
                out = torch.cat([x1, x2], dim=1)
                log_det = x.new_zeros(x.size(0))
        elif self.flow_type == 'affine':
            if not reverse:
                out = self.net(x1, init=init)
                outs = out.chunk(2, dim=1)
                shift = outs[0]
                log_scale = self.rescale * torch.tanh(self.scale * outs[1] + self.shift) + self.reshift
                x2 = torch.exp(log_scale) * x2 + shift
                out = torch.cat([x1, x2], dim=1)
                log_det = log_scale.sum([1, 2, 3])
            else:
                out = self.net(x1)
                outs = out.chunk(2, dim=1)
                shift = outs[0]
                log_scale = self.rescale * torch.tanh(self.scale * outs[1] + self.shift) + self.reshift
                x2 = torch.exp(-log_scale) * (x2 - shift)
                out = torch.cat([x1, x2], dim=1)
                log_det = log_scale.sum([1, 2, 3]).mul(-1)
        elif self.flow_type == 'matrixexp':
            if not reverse:
                if self.x2_channels <= self.max_out:
                    out = self.net(x1, init=init).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight = torch.cat(outs[1:], dim=2).permute(0, 3, 4, 1, 2)
                    weight = self.rescale * torch.tanh(self.scale * weight + self.shift) + self.reshift
                    x2 = x2.unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(expm(weight), x2).permute(0, 3, 4, 1, 2).squeeze(2) + shift
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight, dim1=-2, dim2=-1).sum([1, 2, 3])
                else:
                    out = self.net(x1, init=init).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight1 = torch.cat(outs[1:self.k + 1], dim=2).permute(0, 3, 4, 2, 1)
                    weight2 = torch.cat(outs[self.k + 1:2 * self.k + 1], dim=2).permute(0, 3, 4, 1, 2)
                    weight1 = self.rescale * torch.tanh(self.scale * weight1 + self.shift) + self.reshift + eps
                    weight2 = self.rescale * torch.tanh(self.scale * weight2 + self.shift) + self.reshift + eps
                    weight3 = torch.matmul(weight1, weight2)
                    weight = torch.eye(self.x2_channels, device=x.device) + torch.matmul(
                        torch.matmul(weight2, series(weight3)), weight1)
                    x2 = x2.unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(weight, x2).permute(0, 3, 4, 1, 2).squeeze(2) + shift
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight3, dim1=-2, dim2=-1).sum([1, 2, 3])
            else:
                if self.x2_channels <= self.max_out:
                    out = self.net(x1).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight = torch.cat(outs[1:], dim=2).permute(0, 3, 4, 1, 2)
                    weight = self.rescale * torch.tanh(self.scale * weight + self.shift) + self.reshift
                    x2 = (x2 - shift).unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(expm(-weight), x2).permute(0, 3, 4, 1, 2).squeeze(2)
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight, dim1=-2, dim2=-1).sum([1, 2, 3]).mul(-1)
                else:
                    out = self.net(x1).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight1 = torch.cat(outs[1:self.k + 1], dim=2).permute(0, 3, 4, 2, 1)
                    weight2 = torch.cat(outs[self.k + 1:2 * self.k + 1], dim=2).permute(0, 3, 4, 1, 2)
                    weight1 = self.rescale * torch.tanh(self.scale * weight1 + self.shift) + self.reshift + eps
                    weight2 = self.rescale * torch.tanh(self.scale * weight2 + self.shift) + self.reshift + eps
                    weight3 = torch.matmul(weight1, weight2)
                    weight = torch.eye(self.x2_channels, device=x.device) - torch.matmul(
                        torch.matmul(weight2, series(-weight3)), weight1)
                    x2 = (x2 - shift).unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(weight, x2).permute(0, 3, 4, 1, 2).squeeze(2)
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight3, dim1=-2, dim2=-1).sum([1, 2, 3]).mul(-1)
        else:
            raise ValueError('wrong flow type')

        return out, log_det

    def extra_repr(self):
        return 'in_channels={}, hidden_channels={}, out_channels={},flow_type={}'.format(self.in_channels,
                                                                                         self.hidden_channels,
                                                                                         self.in_channels,
                                                                                         self.flow_type)