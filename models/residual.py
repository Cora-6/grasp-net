"""The byte-equivalent residual block used by the frozen dual branch."""
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels))

    def forward(self, x):
        return x + self.layers(x)
