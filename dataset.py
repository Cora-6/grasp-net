"""GraspNet planar RGB-D adapter; reuse the project's augmentation pipeline."""
import os
import tempfile
from pathlib import Path

from grasp import Cornell, box, targets
import hashlib, inspect
TARGET_CODE_HASH = hashlib.sha256(inspect.getsource(targets).encode()).hexdigest()
import cv2
import numpy as np


def load_rectangles(path):
    a = np.load(path, allow_pickle=False)
    if a.ndim != 2 or a.shape[1] != 7:
        raise ValueError('Expected N x 7 official rectangle labels')
    # Official representation: cx, cy, open_x, open_y, height, score, object_id.
    # Use supplied rectangle labels, as official loadGrasp(format='rect') does.
    # Do not interpret score as the raw friction coefficient or add a guessed filter.
    v = a[:, 2:4] - a[:, :2]
    keep = np.isfinite(a).all(1) & (np.linalg.norm(v, axis=1) > 0) & (a[:, 4] > 0)
    a, v = a[keep], v[keep]
    normal = np.stack((-v[:, 1], v[:, 0]), axis=1)
    normal = normal / np.linalg.norm(v, axis=1)[:, None] * (a[:, 4:5] / 2)
    center = a[:, :2]
    return np.stack((center-v-normal, center+v-normal, center+v+normal, center-v+normal), axis=1).astype(np.float32)


class GraspNet(Cornell):
    def target_for(self, i, rs, rotation):
        # Dense GraspNet labels can contain tens of thousands of rectangles.
        # Cache exact target maps; sensor corruptions never change labels.
        row = self.rows[i]
        p = self.root / row['sample']
        label = p.parent / 'rect' / (p.name + '.npy')
        signature = str(label.resolve()) + str(label.stat().st_mtime_ns) + str(label.stat().st_size)
        signature += "vectorized_rect_v2" + TARGET_CODE_HASH + str(self.n) + str(rotation)
        key = hashlib.sha256(signature.encode()).hexdigest()
        cache = getattr(self, '_target_cache', None)
        if cache is None:
            cache = self._target_cache = {}
        if key not in cache:
            folder = Path(os.environ.get('GRASPNET_TARGET_CACHE', self.root / '_target_cache'))
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (key + '.npz')
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    target = saved['target']
            else:
                target = targets(rs, self.n)
                with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as stream:
                    tmp = Path(stream.name)
                try:
                    np.savez_compressed(tmp, target=target)
                    tmp.replace(path)
                finally:
                    if tmp.exists():
                        tmp.unlink()
            if len(cache) >= 128:
                cache.pop(next(iter(cache)))
            cache[key] = target
        return cache[key]

    def base(self, i):
        if i not in self.cache:
            row = self.rows[i]
            p = self.root / row['sample']
            frame = p.name
            camera = p.parent
            rgb_path = camera / 'rgb' / (frame + '.png')
            dp = camera / 'depth' / (frame + '.png')
            rgb = cv2.imread(str(rgb_path))
            d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            if rgb is None or d is None:
                raise FileNotFoundError('Missing RGB/depth for ' + str(p))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            d = d.astype(np.float32) / 1000.0  # Official raw depth PNG: millimetres.
            d[~np.isfinite(d) | (d < 0)] = 0
            if rgb.shape[:2] != d.shape:
                raise ValueError('RGB/depth dimensions differ')
            rs = load_rectangles(camera / 'rect' / (frame + '.npy'))
            h, w = d.shape
            scale = self.n / max(h, w)
            nw, nh = round(w * scale), round(h * scale)
            ox, oy = (self.n - nw)//2, (self.n - nh)//2
            rgb2 = np.zeros((self.n, self.n, 3), np.uint8)
            d2 = np.zeros((self.n, self.n), np.float32)
            rgb2[oy:oy+nh, ox:ox+nw] = cv2.resize(rgb, (nw, nh))
            d2[oy:oy+nh, ox:ox+nw] = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_NEAREST)
            # Bound memory even when all 48,640 single-camera frames are used.
            if len(self.cache) >= 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = (rgb2, d2, rs * scale + np.float32([ox, oy]))
        return tuple(x.copy() for x in self.cache[i])
