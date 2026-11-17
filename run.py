import sys
from pathlib import Path

import argparse, csv, json, time, platform
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grasp import *
from dataset import GraspNet as Cornell
from models import build_model, model_loss, decode_output, metadata, NAMES, LOSS_PROFILES


def device_of(name):
    if name != "auto":
        return torch.device(name)
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


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


def train(a):
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise ValueError(f"Training output is not empty: {out}; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    device = device_of(a.device)
    torch.set_num_threads(a.threads)
    model = build_model(a.model, a.loss_profile).to(device)
    if a.model == "grconvnet" and a.size % 4:
        raise ValueError("GR-ConvNet --size must be divisible by 4")
    initial_hash = None
    if a.init_checkpoint:
        initial = torch.load(a.init_checkpoint, map_location="cpu", weights_only=False)
        initial_cfg = initial["config"]
        for key, value in {"model_name": a.model, "loss_profile": a.loss_profile,
                           "size": a.size, "split_sha256": hashlib.sha256(Path(a.split).read_bytes()).hexdigest()}.items():
            if initial_cfg.get(key, "original" if key == "loss_profile" else None) != value:
                raise ValueError(f"Initialization mismatch: {key}")
        model.load_state_dict(initial["model"])
        initial_hash = hashlib.sha256(Path(a.init_checkpoint).read_bytes()).hexdigest()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=a.lr)
    tr = Cornell(a.data, a.split, "train", a.size, a.variant, True, seed=a.seed)
    va = Cornell(a.data, a.split, "val", a.size, seed=a.seed)
    va_missing = Cornell(a.data, a.split, "val", a.size, condition="missing", level=1, seed=42)
    dl = torch.utils.data.DataLoader(
        tr,
        batch_size=a.batch,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(a.seed),
    )
    info = vars(a).copy()
    info.update(metadata(model))
    info.update(dataset="GraspNet-1Billion", metric="project_planar_iou_not_official_3d_AP", grouping="scene")
    info.update(
        init_checkpoint_sha256=initial_hash,
        device=str(device),
        torch=torch.__version__,
        platform=platform.platform(),
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        split_sha256=hashlib.sha256(Path(a.split).read_bytes()).hexdigest(),
    )
    (out / "config.json").write_text(json.dumps(info, indent=2))
    best = (-1, float("-inf"))
    history = []
    start_epoch = 0
    elapsed_before = 0.0
    if a.resume:
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if "optimizer" not in saved:
            raise ValueError("Legacy checkpoint has no optimizer state; exact resume unavailable")
        for key in ("model_name", "loss_profile", "split_sha256", "size", "batch", "seed", "variant", "lr", "device"):
            if key == "device" and a.resume_on_mps and saved["config"].get(key) == "cpu" and device.type == "mps":
                continue
            if saved["config"].get(key) != info.get(key):
                raise ValueError("Resume configuration mismatch: " + key)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        history = saved["history"]
        best = tuple(saved["best_score"])
        start_epoch = saved["epoch"]
        elapsed_before = saved["elapsed_seconds"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        dl.generator.set_state(saved["rng"]["loader"])
        if saved["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        if saved["rng"].get("mps") is not None:
            torch.mps.set_rng_state(saved["rng"]["mps"])
    start = time.time()
    for epoch in range(start_epoch, a.epochs):
        model.train()
        losses = []
        for batch_index, (x, y) in enumerate(dl):
            opt.zero_grad()
            loss = model_loss(model, model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if batch_index % 10 == 0 or batch_index + 1 == len(dl):
                progress = dict(epoch=epoch+1, epochs=a.epochs, batch=batch_index+1, batches=len(dl), stage="training", model=a.model, variant=a.variant)
                (out / "progress.json").write_text(json.dumps(progress))
                print(json.dumps(dict(progress=progress)), flush=True)
        rec = evaluate(model, va, device)
        acc = float(np.mean([r["success"] for r in rec]))
        val_loss = float(np.mean([r["loss"] for r in rec]))
        missing_acc = None
        if a.selection == "clean_and_light_missing":
            va_missing.cache = va.cache
            rec_missing = evaluate(model, va_missing, device)
            missing_acc = float(np.mean([r["success"] for r in rec_missing]))
        selection_score = acc if missing_acc is None else (acc + missing_acc) / 2
        score = (selection_score, -val_loss)
        entry = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": acc,
            "val_loss": val_loss,
            "val_light_missing_accuracy": missing_acc,
            "selection_score": selection_score,
            "elapsed_seconds": elapsed_before + time.time() - start,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        improved = score > best
        if improved:
            best = score
        state = {"model": model.state_dict(), "config": info, "epoch": epoch + 1,
                 "optimizer": opt.state_dict(), "history": history, "best_score": best,
                 "elapsed_seconds": elapsed_before + time.time() - start,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(), "loader": dl.generator.get_state(),
                         "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                         "mps": torch.mps.get_rng_state() if device.type == "mps" else None}}
        def save_atomic(value, path):
            temp = path.with_suffix(path.suffix + ".tmp")
            torch.save(value, temp)
            temp.replace(path)
        save_atomic(state, out / "last.pt")
        if improved:
            save_atomic(state, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))



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
        ds = Cornell(
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "evaluate"])
    p.add_argument("--data", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--no-visuals", action="store_true")
    p.add_argument(
        "--variant",
        choices=[
            "baseline",
            "sensor",
            "dropout",
            "combined",
            "noise",
            "missing",
            "occlusion",
            "mild_missing",
            "mild_gaussian", "mild_irregular", "mild_smooth", "mild_edge", "mild_occlusion",
        ],
        default="baseline",
    )
    p.add_argument("--model", choices=NAMES, default="grconvnet", help="Training architecture; evaluation reads checkpoint metadata")
    p.add_argument("--loss-profile", choices=LOSS_PROFILES, default="foreground_balanced")
    p.add_argument("--init-checkpoint")
    p.add_argument("--resume", action="store_true", help="Restore model, Adam, RNG and loader at completed epoch boundary")
    p.add_argument("--resume-on-mps", action="store_true", help="Permit migrating a CPU checkpoint to MPS")
    p.add_argument("--selection", choices=["clean", "clean_and_light_missing"], default="clean")
    p.add_argument("--test-suite", choices=["legacy", "extended"], default="legacy")
    p.add_argument("--part", choices=["val", "test_seen", "test_similar", "test_novel"], default="test_novel")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    if a.command == "train":
        train(a)
    else:
        test(a)
