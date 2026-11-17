"""Per-image scoring and the fixed corruption evaluation suite."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets.graspnet import GraspNetDataset
from datasets.corruptions import SEVERITY, NEW_KINDS
from evaluation.geometry import match
from evaluation.decoder import decode_output
from losses import model_loss
from models.registry import build_model
from models.protocols import metadata
from utils.runtime import device_of

@torch.no_grad()
def evaluate(model, ds, device, visual=None):
    model.eval()
    records = []
    for i, row in enumerate(ds.rows):
        x, y, rs, rgb, d = ds.sample(i)
        output = model(torch.from_numpy(x[None]).to(device))
        pred, q = decode_output(model, output[0])
        ok, iou = match(pred, rs)
        loss = float(model_loss(model, output, torch.from_numpy(y[None]).to(device)).cpu())
        records.append(
            {
                "sample": row["sample"],
                "object_id": row["object_id"],
                "success": int(ok),
                "best_iou_angle_gated": iou,
                "loss": loss,
            }
        )
        if visual and i < 6:
            fig, ax = plt.subplots(1, 3, figsize=(10, 3))
            ax[0].imshow(rgb)
            ax[1].imshow(d, cmap="viridis")
            ax[2].imshow(q, vmin=0, vmax=1, cmap="magma")
            for r in rs:
                p = np.vstack([r, r[0]])
                ax[0].plot(p[:, 0], p[:, 1], "g-", alpha=0.4)
            p = np.vstack([pred, pred[0]])
            ax[0].plot(p[:, 0], p[:, 1], "r-", lw=2)
            ax[0].set_title(f'{row["sample"]}: {"success" if ok else "failure"}')
            ax[1].set_title("Depth (m)")
            ax[2].set_title("Grasp quality")
            for a in ax:
                a.axis("off")
            fig.tight_layout()
            fig.savefig(visual / f"{i:02d}.png", dpi=120)
            plt.close(fig)
    return records

def test(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = device_of(a.device)
    torch.set_num_threads(a.threads)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ck["config"].get("model_name", "tiny_unet"), ck["config"].get("loss_profile", "original")).to(device)
    current = metadata(model)
    for key in ("output_protocol", "upstream_commit", "upstream_source_sha256", "loss_profile"):
        if key in ck["config"] and ck["config"][key] != current.get(key):
            raise ValueError(f"Checkpoint protocol/source mismatch: {key}")
    model.load_state_dict(ck["model"])
    split_hash = hashlib.sha256(Path(a.split).read_bytes()).hexdigest()
    if split_hash != ck["config"]["split_sha256"]:
        raise ValueError("Checkpoint and split differ")
    results = []
    details = []
    cache = {}
    conditions = (
        [("clean", 0)]
        + [(k, l) for k in SEVERITY for l in (1, 2, 3)]
        + [("dropout", 1)]
    )
    if a.test_suite == "extended":
        conditions += [(k,l) for k in NEW_KINDS for l in (1,2,3)]
    for kind, level in conditions:
        ds = GraspNetDataset(
            a.data,
            a.split,
            a.part,
            ck["config"]["size"],
            condition=kind,
            level=level,
            seed=a.eval_seed,
        )
        ds.cache = cache
        vis = None if a.no_visuals else out / f"{kind}_{level}"
        if vis is not None:
            vis.mkdir(exist_ok=True)
        rec = evaluate(model, ds, device, vis)
        acc = float(np.mean([r["success"] for r in rec]))
        objs = sorted({r["object_id"] for r in rec})
        by_obj = {o: [r["success"] for r in rec if r["object_id"] == o] for o in objs}
        rng = np.random.default_rng(a.eval_seed)
        boots = [
            np.mean(
                [
                    s
                    for o in rng.choice(objs, len(objs), replace=True)
                    for s in by_obj[o]
                ]
            )
            for _ in range(1000)
        ]
        entry = {
            "condition": kind,
            "severity": level,
            "accuracy": acc,
            "n_images": len(rec),
            "n_groups": len(objs),
            "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)),
        }
        results.append(entry)
        details.extend([dict(r, condition=kind, severity=level) for r in rec])
        print(entry, flush=True)
    clean = results[0]["accuracy"]
    for r in results:
        r["retention"] = r["accuracy"] / clean if clean else None
    (out / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "evaluation_seed": a.eval_seed,
                "part": a.part,
                "loss_profile": ck["config"].get("loss_profile", "original"),
                "model_name": ck["config"].get("model_name", "tiny_unet"),
                "output_protocol": ck["config"].get("output_protocol", "legacy_logits_v1"),
                "checkpoint_sha256": hashlib.sha256(
                    Path(a.checkpoint).read_bytes()
                ).hexdigest(),
                "split_sha256": split_hash,
                "checkpoint_epoch": ck["epoch"],
                "device": str(device),
                "conditions": len(conditions),
                "test_suite": a.test_suite,
            },
            indent=2,
        )
    )
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "predictions.json").write_text(json.dumps(details, indent=2))
    with open(out / "metrics.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, kind in zip(axes, SEVERITY):
        rr = [results[0]] + [r for r in results if r["condition"] == kind]
        ax.plot(range(4), [r["accuracy"] for r in rr], "o-")
        ax.set(
            title=kind,
            xlabel="severity",
            ylabel="grasp accuracy",
            ylim=(0, 1),
            xticks=range(4),
        )
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=160)
    plt.close(fig)
