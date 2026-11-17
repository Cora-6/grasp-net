"""Train the GraspNet main comparison and matched fusion ablation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ggcnn", "grconvnet", "dual_concat", "early_concat")
TEST_PARTS = ("test_seen", "test_similar", "test_novel")


def label_path(data, row):
    frame = data / row["sample"]
    return frame.parent / "rect" / (frame.name + ".npy")


def one_per_scene(rows):
    chosen = {}
    for row in rows:
        chosen.setdefault(row["scene_id"], row)
    return list(chosen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "split", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--pilot", action="store_true", help="One frame per labeled scene, one seed, one epoch")
    modes.add_argument("--labeled-seen-only", action="store_true",
                       help="Train on the full split and evaluate on currently labeled seen frames only")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--research-plan", action="store_true",
                        help="Use depth-only GG-CNN and native baseline losses")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--reuse-grconvnet", type=Path, help="Directory containing completed baseline_s* runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-ggcnn2", action="store_true", help="Add the four-channel GG-CNN2 baseline")
    parser.add_argument("--models", nargs="+", choices=(*MODELS, "ggcnn_depth", "ggcnn2", "ggcnn2_depth"),
                        help="Run only selected models, in a separate output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    split = json.loads(args.split.read_text())
    parts = ("train", "val", *TEST_PARTS)
    coverage = {part: [row for row in split[part] if label_path(data, row).is_file()]
                for part in parts}
    print("Rectangle coverage:", {p: (len(coverage[p]), len(split[p])) for p in parts}, flush=True)
    if args.pilot:
        for part in parts:
            split[part] = one_per_scene(coverage[part])
        test_parts = [part for part in TEST_PARTS if split[part]]
    elif args.labeled_seen_only:
        if any(len(coverage[p]) != len(split[p]) for p in ("train", "val")):
            parser.error("All training and validation labels are required")
        split["test_seen"] = coverage["test_seen"]
        for part in ("test_similar", "test_novel"):
            split[part] = []
        test_parts = ["test_seen"] if split["test_seen"] else []
    else:
        missing = [part for part in parts if len(coverage[part]) != len(split[part])]
        if missing:
            parser.error("Missing rectangle annotations in " + ", ".join(missing))
        test_parts = TEST_PARTS
    if not (split["train"] and split["val"] and test_parts):
        parser.error("Training, validation and at least one usable test group are required")
    for part in ("train", "val", *test_parts):
        for row in split[part]:
            frame = data / row["sample"]
            for folder in ("rgb", "depth"):
                if not (frame.parent / folder / (frame.name + ".png")).is_file():
                    parser.error("Missing " + folder + " for " + str(frame))
    out.mkdir(parents=True, exist_ok=args.resume)
    epochs, seeds = (1, (42,)) if args.pilot else (args.epochs, tuple(args.seeds))
    models = ("ggcnn_depth", *MODELS[1:]) if args.research_plan else MODELS
    if args.include_ggcnn2:
        models = (*models, "ggcnn2")
    if args.models:
        models = tuple(dict.fromkeys(args.models))
    loss = "original" if args.research_plan else "foreground_balanced"
    plan = dict(pilot=args.pilot, labeled_seen_only=args.labeled_seen_only,
                main_models=[m for m in models if m != "early_concat"],
                ablation=[m for m in models if m in ("dual_concat", "early_concat")],
                test_parts=test_parts, epochs=epochs, seeds=seeds, size=300,
                counts={p: len(split[p]) for p in parts}, loss=loss,
                selection="clean", data=str(data), eval_seed=42, test_suite="legacy",
                training_variant="baseline", batch=8, learning_rate=0.001, device=args.device)
    split["protocol"].update(pilot=args.pilot, labeled_seen_only=args.labeled_seen_only)
    split_path = out / "split.json"
    if split_path.exists():
        if json.loads(split_path.read_text()) != split:
            parser.error("Existing split differs from the requested split")
    else:
        split_path.write_text(json.dumps(split, indent=2))
    reused = {}
    if args.reuse_grconvnet:
        for seed in seeds:
            folder = args.reuse_grconvnet.resolve() / f"baseline_s{seed}"
            if not (folder / "complete.json").is_file():
                continue
            config = json.loads((folder / "config.json").read_text())
            source_split = Path(config["split"])
            original = json.loads(source_split.read_text())
            checkpoint = folder / "best.pt"
            if (not checkpoint.is_file() or config["model_name"] != "grconvnet"
                    or config["variant"] != "baseline" or config["seed"] != seed
                    or any(original[p] != split[p] for p in parts)):
                parser.error(f"Incompatible GR-ConvNet checkpoint: {folder}")
            reused[f"grconvnet_s{seed}"] = dict(checkpoint=str(checkpoint), split=str(source_split),
                                               size=config["size"], loss=config["loss_profile"])
        if not reused:
            parser.error("No completed GR-ConvNet baseline found")
        plan["reused"] = reused
        plan["excluded"] = [f"grconvnet_s{seed}" for seed in seeds if f"grconvnet_s{seed}" not in reused]
    (out / "comparison_protocol.json").write_text(json.dumps(plan, indent=2))
    env = os.environ.copy()
    env.setdefault("GRASPNET_TARGET_CACHE", str(out.parent / "targets_300"))
    common = ["--data", str(data), "--split", str(split_path), "--device", args.device]
    jobs = [(model, seed, out / f"{model}_s{seed}") for seed in seeds for model in models]

    def run(command, params, log, base=common):
        print(command, log, flush=True)
        with log.open("a") as stream:
            subprocess.run([sys.executable, "-u", str(ROOT / "run.py"), command, *base, *params],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    for model, seed, dest in jobs:
        if args.reuse_grconvnet and model == "grconvnet":
            continue
        history = dest / "history.json"
        if history.exists() and len(json.loads(history.read_text())) >= epochs:
            continue
        resume = ["--resume"] if (dest / "last.pt").is_file() else []
        if resume and args.device == "mps":
            resume.append("--resume-on-mps")
        run("train", ["--out", str(dest), "--model", model, "--seed", str(seed),
                      "--size", "300", "--epochs", str(epochs), "--batch", "8",
                      "--loss-profile", loss, *resume], out / f"{dest.name}_train.log")

    if args.train_only:
        return

    summary = {}
    for _, _, dest in jobs:
        if dest.name in plan.get("excluded", []):
            continue
        summary[dest.name] = {}
        existing = reused.get(dest.name)
        checkpoint = existing["checkpoint"] if existing else str(dest / "best.pt")
        eval_args = ["--data", str(data), "--split", existing["split"], "--device", args.device] if existing else common
        for part in test_parts:
            result = dest / part
            run("evaluate", ["--out", str(result), "--checkpoint", checkpoint,
                             "--part", part, "--no-visuals"], out / f"{dest.name}_{part}.log", eval_args)
            summary[dest.name][part] = json.loads((result / "metrics.json").read_text())
    (out / "model_comparison.json").write_text(json.dumps(dict(protocol=plan, results=summary), indent=2))


if __name__ == "__main__":
    main()
