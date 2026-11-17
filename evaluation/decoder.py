"""Decode dense model predictions into one planar grasp rectangle."""
import numpy as np
from scipy.ndimage import gaussian_filter
from models.protocols import model_name
from evaluation.geometry import box
from legacy.tiny_unet import decode

def decode_output(model, output):
    if model_name(model) == 'tiny_unet':
        return decode(output)
    o = output.detach().cpu().numpy()
    # Linear heads: no sigmoid. scipy mode nearest matches skimage gaussian default.
    q = gaussian_filter(o[0], 2, mode='nearest')
    angle = gaussian_filter(np.arctan2(o[2], o[1]) / 2, 2, mode='nearest')
    width = gaussian_filter(o[3] * (o.shape[-1] / 2), 1, mode='nearest')
    y, x = np.unravel_index(q.argmax(), q.shape)
    w = max(2.0, float(width[y, x]))
    return box([x, y], float(angle[y, x]), w, w / 2), q
