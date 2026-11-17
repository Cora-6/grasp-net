"""Original GG-CNN with four RGB-D input channels."""
import torch
from torch import nn

from third_party.ggcnn.ggcnn import GGCNN


class GGCNNAdapter(nn.Module):
    architecture = "ggcnn"

    def __init__(self, depth_only=False):
        super().__init__()
        self.depth_only = depth_only
        if depth_only:
            self.architecture = "ggcnn_depth"
        self.network = GGCNN(input_channels=1 if depth_only else 4)

    def forward(self, x):
        inputs = x[:, 3:4] if self.depth_only else x[:, [3, 0, 1, 2]]
        outputs = self.network(inputs.contiguous())
        if outputs[0].shape[-2:] != x.shape[-2:]:
            raise ValueError("Use 300-pixel inputs for original GG-CNN")
        return torch.cat(outputs, dim=1)

# Historical class name retained for external callers.
GGCNNRGBD = GGCNNAdapter
