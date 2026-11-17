"""Backward-compatible CLI: python run.py train|evaluate [options]."""
from engine.cli import main
from engine.trainer import train
from evaluation.evaluator import evaluate, test
from utils.runtime import device_of

if __name__ == '__main__':
    main()
