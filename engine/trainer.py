"""Training, validation-based checkpoint selection and exact resume."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import platform
import random
import time
from datasets.graspnet import GraspNetDataset
from evaluation.evaluator import evaluate
from models.registry import build_model
from models.protocols import metadata
from losses import model_loss
from utils.runtime import seed_all, device_of

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
    tr = GraspNetDataset(a.data, a.split, "train", a.size, a.variant, True, seed=a.seed)
    va = GraspNetDataset(a.data, a.split, "val", a.size, seed=a.seed)
    va_missing = GraspNetDataset(a.data, a.split, "val", a.size, condition="missing", level=1, seed=42)
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
