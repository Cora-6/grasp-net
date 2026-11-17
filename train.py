"""Train a model; use --config for a JSON experiment preset."""
import sys
from engine.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
