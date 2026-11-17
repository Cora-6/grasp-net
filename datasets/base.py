"""Shared RGB-D sampling, augmentation and normalization. Internal order: R/G/B/D."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from datasets.corruptions import corrupt, LIGHT_VARIANTS
from datasets.targets import targets

class RGBDGraspDataset(torch.utils.data.Dataset):
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
        """Return RGB, metre depth and rectangles before training transforms."""
        raise NotImplementedError

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
