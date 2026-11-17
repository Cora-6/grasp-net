"""Output adapter around the author's original GR-ConvNet. No pretrained weights."""
import torch
from torch import nn
from third_party.grconvnet.grconvnet import GenerativeResnet


class GRConvNet(nn.Module):
    architecture = 'grconvnet'

    def __init__(self):
        super().__init__()
        self.network = GenerativeResnet(input_channels=4, dropout=False, prob=0.0)

    def forward(self, x):
        if x.shape[-1] != x.shape[-2] or x.shape[-1] % 4:
            raise ValueError('GR-ConvNet requires square input with size divisible by 4')
        # Dataset uses RGBD; author implementation expects depth followed by RGB.
        d_rgb = torch.cat((x[:, 3:4], x[:, :3]), dim=1)
        return torch.cat(self.network(d_rgb), dim=1)
