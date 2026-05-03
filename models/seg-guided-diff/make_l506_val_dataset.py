#!/usr/bin/env python3
"""
Create a copied dataset where patient L506 is the validation split.

The source dataset is left untouched. The script copies the full dataset root,
then updates every train/val split pair it finds under the copy:

1. Move the current validation files into train.
2. Move every train file whose filename starts with L506 into val.

Default input/output:
  <script directory>/dataset -> <script directory>/dataset_l506_val
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a dataset and make L506 the validation patient."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=SCRIPT_DIR / "dataset",
        help="Source dataset root. Defaults to the dataset directory next to this script.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=SCRIPT_DIR / "dataset_l506_val",
        help="Destination dataset root to create next to this script by default.",
    )
    parser.add_argument(
        "--patient-prefix",
        default="L506",
        help="Filename prefix to use as the new validation patient. Defaults to L506.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the destination first if it already exists.",
    )
    return parser.parse_args()


def find_split_roots(dataset_root: Path) -> list[Path]:
    """Return directories that directly contain both train and val subdirs."""
    split_roots: list[Path] = []
    for candidate in [dataset_root, *dataset_root.rglob("*")]:
        if not candidate.is_dir():
            continue
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            split_roots.append(candidate)
    return sorted(split_roots)


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def move_files(files: list[Path], source_split: Path, target_split: Path) -> int:
    moved = 0
    for source_path in files:
        relative_path = source_path.relative_to(source_split)
        target_path = target_split / relative_path

        if target_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {target_path}"
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        moved += 1

    return moved


def make_l506_val_split(dataset_root: Path, patient_prefix: str) -> None:
    split_roots = find_split_roots(dataset_root)
    if not split_roots:
        raise RuntimeError(f"No train/val split pairs found under {dataset_root}")

    total_old_val_to_train = 0
    total_patient_to_val = 0

    for split_root in split_roots:
        train_dir = split_root / "train"
        val_dir = split_root / "val"

        old_val_files = iter_files(val_dir)
        total_old_val_to_train += move_files(old_val_files, val_dir, train_dir)

        patient_train_files = [
            path
            for path in iter_files(train_dir)
            if path.name.startswith(patient_prefix)
        ]
        total_patient_to_val += move_files(patient_train_files, train_dir, val_dir)

    print(f"Updated dataset: {dataset_root}")
    print(f"Moved old validation files to train: {total_old_val_to_train}")
    print(f"Moved {patient_prefix} files to val: {total_patient_to_val}")


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {src}")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}. "
                "Choose another --dst or pass --overwrite."
            )
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    make_l506_val_split(dst, args.patient_prefix)


if __name__ == "__main__":
    main()
