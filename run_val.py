from __future__ import annotations

import argparse
import os
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

from config import load_config
from validate import run_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--ckpt", "--checkpoint", dest="checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--fold", type=int, default=None, help="Override split_id from config.")
    parser.add_argument("--device", default=None, help="Override device from config.")
    return parser.parse_args()


def infer_output_dir(checkpoint_path: str | Path) -> Path | None:
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "validation"
    return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device

    fold = args.fold if args.fold is not None else int(config.get("split_id", 0))
    output_dir = infer_output_dir(args.checkpoint)
    summary = run_validation(config, args.checkpoint, fold=fold, output_dir=output_dir)
    print(f"Mean validation Dice: {summary['foreground_mean']['Dice']:.4f}")


if __name__ == "__main__":
    main()