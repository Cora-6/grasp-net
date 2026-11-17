"""Planar rectangles: (x, y), clockwise image angles, IoU/angle matching."""
import cv2
import numpy as np

def params(r):
    v = r[1] - r[0]
    return (
        r.mean(0),
        float(np.arctan2(v[1], v[0])),
        float(np.linalg.norm(v)),
        float(np.linalg.norm(r[2] - r[1])),
    )

def box(center, angle, width, height):
    u = np.array([np.cos(angle), np.sin(angle)]) * width / 2
    v = np.array([-np.sin(angle), np.cos(angle)]) * height / 2
    c = np.asarray(center)
    return np.float32([c - u - v, c + u - v, c + u + v, c - u + v])

def match(pred, truth):
    a = params(pred)[1]
    best = 0.0
    for r in truth:
        delta = abs((a - params(r)[1] + np.pi / 2) % np.pi - np.pi / 2)
        if delta >= np.pi / 6:
            continue
        inter, _ = cv2.intersectConvexConvex(
            pred.astype("float32"), r.astype("float32")
        )
        union = abs(cv2.contourArea(pred)) + abs(cv2.contourArea(r)) - inter
        best = max(best, float(inter / max(union, 1e-8)))
    return best > 0.25, best
