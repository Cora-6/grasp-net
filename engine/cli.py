"""Shared CLI for the train/evaluate entry points."""
import argparse
from engine.trainer import train
from evaluation.evaluator import test
from models.registry import NAMES
from losses import LOSS_PROFILES
from utils.config import parse_with_config

def main(argv=None):
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
    a = parse_with_config(p, argv)
    if a.command == "train":
        train(a)
    else:
        test(a)
