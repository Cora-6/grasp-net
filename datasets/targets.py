"""Dense quality, double-angle and width supervision maps."""
import cv2
import numpy as np
from evaluation.geometry import box, params

def targets(rs, n):
    # Last annotation wins on overlaps, exactly as in the original loop.
    indices = np.zeros((n, n), np.int32)
    values = np.zeros((len(rs) + 1, 4), np.float32)
    for index, r in enumerate(rs, 1):
        c, a, w, h = params(r)
        polygon = np.round(box(c, a, w / 3, h)).astype("int32")
        cv2.fillConvexPoly(indices, polygon, index)
        values[index] = (1, np.cos(2 * a), np.sin(2 * a), min(w / n, 1))
    return values[indices].transpose(2, 0, 1).copy()
