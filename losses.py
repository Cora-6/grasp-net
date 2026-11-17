"""Training objectives, separate from architecture definitions."""
import torch
from models.protocols import model_name
from legacy.tiny_unet import loss_fn

LOSS_PROFILES = ("original", "foreground_balanced")

def grconv_targets(y):
    """Upstream uses cos(0)=1 on background; width is normalized by half size."""
    target = y.clone()
    target[:, 1] = torch.where(y[:, 0] > 0, y[:, 1], torch.ones_like(y[:, 1]))
    target[:, 3] = (y[:, 3] * 2).clamp(0, 1)
    return target

def model_loss(model, output, target):
    if model_name(model) == 'tiny_unet':
        return loss_fn(output, target)
    target = grconv_targets(target)
    if getattr(model, 'loss_profile', 'original') == 'foreground_balanced':
        # Project adaptation: equal foreground/background quality weight;
        # angle/width matter only where a grasp is annotated.
        mask = target[:, 0:1]
        quality = torch.nn.functional.smooth_l1_loss(
            output[:, 0:1].contiguous(), mask.contiguous(), reduction='none')
        fg = mask.sum().clamp_min(1)
        bg = (1 - mask).sum().clamp_min(1)
        loss = .5 * ((quality * mask).sum() / fg + (quality * (1-mask)).sum() / bg)
        for i in range(1, 4):
            error = torch.nn.functional.smooth_l1_loss(
                output[:, i:i+1].contiguous(), target[:, i:i+1].contiguous(), reduction='none')
            loss = loss + (error * mask).sum() / fg
        return loss
    if model_name(model) in ('ggcnn', 'ggcnn_depth', 'ggcnn2', 'ggcnn2_depth'):
        return sum(torch.nn.functional.mse_loss(output[:, i:i+1], target[:, i:i+1]) for i in range(4))
    # Exactly the four unmasked Smooth-L1 terms in upstream GraspModel.compute_loss.
    return sum(torch.nn.functional.smooth_l1_loss(output[:, i:i+1].contiguous(), target[:, i:i+1].contiguous()) for i in range(4))
