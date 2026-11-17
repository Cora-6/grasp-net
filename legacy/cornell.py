"""Historical Cornell file adapter, outside the GraspNet main path."""
from pathlib import Path
import cv2
import numpy as np
from datasets.base import RGBDGraspDataset

def rectangles(path):
    a = (
        np.loadtxt(path, dtype=np.float32).reshape(-1, 4, 2) - 1
    )  # Cornell coordinates are 1-based
    return np.array(
        [r for r in a if np.isfinite(r).all() and abs(cv2.contourArea(r)) > 1],
        dtype=np.float32,
    ).reshape(-1, 4, 2)

class Cornell(RGBDGraspDataset):
    def base(self, i):
        if i not in self.cache:
            row = self.rows[i]
            p = self.root / row["sample"]
            rgb = cv2.cvtColor(cv2.imread(str(p) + "r.png"), cv2.COLOR_BGR2RGB)
            dp = Path(str(p) + "d.tiff")
            if not dp.exists():
                dp = Path(str(p) + "d.npy")
            d = (
                np.load(dp)
                if dp.suffix == ".npy"
                else cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            )
            if d is None:
                raise FileNotFoundError(dp)
            d = np.asarray(d, np.float32)
            d[~np.isfinite(d)] = 0
            rs = rectangles(str(p) + "cpos.txt")
            h, w = d.shape
            # Label-independent square letterbox: no ground-truth-dependent crop at test time.
            scale = self.n / max(h, w)
            nw, nh = round(w * scale), round(h * scale)
            ox, oy = (self.n - nw) // 2, (self.n - nh) // 2
            rgb2 = np.zeros((self.n, self.n, 3), np.uint8)
            d2 = np.zeros((self.n, self.n), np.float32)
            rgb2[oy : oy + nh, ox : ox + nw] = cv2.resize(rgb, (nw, nh))
            d2[oy : oy + nh, ox : ox + nw] = cv2.resize(
                d, (nw, nh), interpolation=cv2.INTER_NEAREST
            )
            self.cache[i] = (rgb2, d2, rs * scale + np.float32([ox, oy]))
        return tuple(x.copy() for x in self.cache[i])
