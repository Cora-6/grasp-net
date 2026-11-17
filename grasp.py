"""Dense RGB-D grasp detector. Coordinates throughout: (x,y), clockwise image angle."""

import hashlib, json, random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from scipy.ndimage import gaussian_filter
from augmentations import NEW_KINDS, LIGHT_VARIANTS, depth_corruption


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rectangles(path):
    a = (
        np.loadtxt(path, dtype=np.float32).reshape(-1, 4, 2) - 1
    )  # Cornell coordinates are 1-based
    return np.array(
        [r for r in a if np.isfinite(r).all() and abs(cv2.contourArea(r)) > 1],
        dtype=np.float32,
    ).reshape(-1, 4, 2)


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


# Noise is measured in metres before standardization. Missing values remain zero afterwards.
SEVERITY = {
    "noise": [0, 0.005, 0.015, 0.03],
    "missing": [0, 0.1, 0.3, 0.5],
    "occlusion": [0, 0.1, 0.2, 0.35],
}


def corrupt(rgb, d, kind, level, rng):
    rgb = rgb.copy()
    d = d.copy()
    n = d.shape[0]
    if kind == "noise":
        valid = d > 0
        d[valid] = np.maximum(
            d[valid] + rng.normal(0, SEVERITY[kind][level], valid.sum()), 1e-6
        )
    elif kind == "missing":
        # One contiguous region of missing depth; exact area recorded by protocol.
        side = max(1, round(n * np.sqrt(SEVERITY[kind][level])))
        x, y = rng.integers(0, n - side + 1, 2)
        d[y : y + side, x : x + side] = 0
    elif kind == "occlusion":
        side = max(1, round(n * np.sqrt(SEVERITY[kind][level])))
        x, y = rng.integers(0, n - side + 1, 2)
        rgb[y : y + side, x : x + side] = 0
        d[y : y + side, x : x + side] = 0
    elif kind == "dropout":
        d[:] = 0
    elif kind in NEW_KINDS:
        d = depth_corruption(d, kind, level, rng)
    elif kind != "clean":
        raise ValueError(kind)
    return rgb, d


class Cornell(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        part,
        n=160,
        variant="baseline",
        training=False,
        condition="clean",
        level=0,
        seed=42,
    ):
        self.root = Path(root)
        with open(split) as f:
            self.rows = json.load(f)[part]
        self.n = n
        self.variant = variant
        self.training = training
        self.condition = condition
        self.level = level
        self.seed = seed
        self.cache = {}

    def __len__(self):
        return len(self.rows)

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

    def sample(self, i):
        rgb, d, rs = self.base(i)
        n = self.n
        rotation = 0
        if self.training:
            rng = np.random.default_rng(np.random.randint(2**31))
            k = int(rng.integers(4))
            rotation = k
            rgb = np.rot90(rgb, k).copy()
            d = np.rot90(d, k).copy()
            for _ in range(k):
                rs = np.stack([rs[..., 1], n - 1 - rs[..., 0]], -1)
            rgb = np.clip(rgb.astype(float) * rng.uniform(0.8, 1.2), 0, 255).astype(
                "uint8"
            )
            kind = "clean"
            level = 0
            if (
                self.variant in ("sensor", "combined", "noise", "missing", "occlusion")
                and rng.random() < 0.75
            ):
                kind = (
                    str(rng.choice(["noise", "missing", "occlusion"]))
                    if self.variant in ("sensor", "combined")
                    else self.variant
                )
                level = int(rng.integers(1, 4))
            if self.variant == "mild_occlusion" and rng.random() < 0.25:
                kind, level = "occlusion", 1
            if self.variant == "mild_missing" and rng.random() < 0.25:
                kind, level = "missing", 1
            if self.variant in LIGHT_VARIANTS and rng.random() < 0.25:
                kind, level = LIGHT_VARIANTS[self.variant], 1
            rgb, d = corrupt(rgb, d, kind, level, rng)
            if self.variant in ("dropout", "combined") and rng.random() < 0.25:
                d[:] = 0
        else:
            token = (
                f'{self.seed}:{self.rows[i]["sample"]}:{self.condition}:{self.level}'
            )
            rng = np.random.default_rng(
                int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
            )
            rgb, d = corrupt(rgb, d, self.condition, self.level, rng)
        valid = d > 0
        dn = np.zeros_like(d)
        # Fixed metre scale, no per-corruption renormalization that would hide noise.
        if valid.any():
            dn[valid] = (d[valid] - np.median(d[valid])) / 0.1
        dn = np.clip(dn, -5, 5)
        x = np.concatenate(
            [rgb.astype("float32") / 255 - 0.5, dn[..., None]], -1
        ).transpose(2, 0, 1)
        target = self.target_for(i, rs, rotation) if hasattr(self, "target_for") else targets(rs, n)
        return x.astype("float32"), target, rs, rgb, d

    def __getitem__(self, i):
        x, y, *_ = self.sample(i)
        return torch.from_numpy(x), torch.from_numpy(y)


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
