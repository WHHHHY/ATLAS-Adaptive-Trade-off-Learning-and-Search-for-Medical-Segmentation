from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path("/root/autodl-tmp/hongyang/nnUNet/nnUNet_preprocessed/Dataset002_BTCV")
DEFAULT_TARGET = Path(__file__).resolve().parent / "data" / "Dataset002_BTCV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source BTCV preprocessed folder.")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="Target folder in the minimal repo.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite target if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    target = Path(args.target)
    if not source.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source}")

    if target.exists() and args.overwrite:
        shutil.rmtree(target)
    elif target.exists():
        print(f"Target already exists: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    print(f"Copied BTCV preprocessed data to {target}")


if __name__ == "__main__":
    main()