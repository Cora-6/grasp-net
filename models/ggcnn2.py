"""Author GG-CNN2 with depth-only or RGB-D inputs, trained from scratch."""
import torch
from torch import nn
from third_party.ggcnn2.ggcnn2 import GGCNN2


class GGCNN2Adapter(nn.Module):
    architecture = 'ggcnn2'

    def __init__(self, depth_only=False):
        super().__init__()
        self.depth_only = depth_only
        if depth_only:
            self.architecture = 'ggcnn2_depth'
        self.network = GGCNN2(input_channels=1 if depth_only else 4)

    def forward(self, x):
        if x.shape[-1] != x.shape[-2] or x.shape[-1] % 4:
            raise ValueError('GG-CNN2 requires square input divisible by four')
        inputs = x[:, 3:4] if self.depth_only else torch.cat((x[:, 3:4], x[:, :3]), dim=1)
        return torch.cat(self.network(inputs.contiguous()), dim=1)

# Historical class name retained for external callers.
GGCNN2RGBD = GGCNN2Adapter
