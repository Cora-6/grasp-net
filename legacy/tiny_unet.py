"""Historical tiny U-Net and its original output conventions."""
import torch
from torch import nn
import numpy as np
from scipy.ndimage import gaussian_filter
from evaluation.geometry import box

class Detector(nn.Module):
    def __init__(self):
        super().__init__()

        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(b, b, 3, padding=1),
                nn.ReLU(),
            )

        self.e1 = block(4, 16)
        self.e2 = block(16, 32)
        self.mid = block(32, 64)
        self.d2 = block(96, 32)
        self.d1 = block(48, 16)
        self.head = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(nn.functional.avg_pool2d(a, 2))
        c = self.mid(nn.functional.avg_pool2d(b, 2))
        c = self.d2(
            torch.cat(
                [
                    nn.functional.interpolate(
                        c, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                ],
                1,
            )
        )
        return self.head(
            self.d1(
                torch.cat(
                    [
                        nn.functional.interpolate(
                            c, size=a.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        a,
                    ],
                    1,
                )
            )
        )

def loss_fn(out, y):
    q = nn.functional.binary_cross_entropy_with_logits(
        out[:, 0], y[:, 0], pos_weight=torch.tensor(5.0, device=out.device)
    )
    m = y[:, 0]
    den = m.sum().clamp(min=1)
    angle = (((out[:, 1:3] - y[:, 1:3]) ** 2) * m[:, None]).sum() / den
    width = (
        nn.functional.smooth_l1_loss(out[:, 3].sigmoid(), y[:, 3], reduction="none") * m
    ).sum() / den
    return q + angle + width

def decode(out):
    o = out.detach().cpu()
    q = gaussian_filter(o[0].sigmoid().numpy(), 1)
    y, x = np.unravel_index(q.argmax(), q.shape)
    a = 0.5 * np.arctan2(float(o[2, y, x]), float(o[1, y, x]))
    w = max(2, float(o[3, y, x].sigmoid()) * q.shape[0])
    return box([x, y], a, w, w / 2), q
