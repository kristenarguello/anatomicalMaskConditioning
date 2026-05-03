#!/usr/bin/env python3
"""
Convert TotalSegmentator NIfTI segmentations to per-class binary PNG masks
in the directory structure and with the exact filenames expected by all model
loaders (RED-CNN, CoreDiff, iDDPM, SegGuidedDiff).

Run this after mask_generation.py.

Input:
  segmentations/{root}/{patient}/quarter_1mm/segmentations.nii.gz

Output:
  {mask_dir}/{tissue_class}/{split}/1mm/QD/{original_qd_filename}.png

The output mirrors the PNG dataset layout exactly: for each patient, slice z
of the NIfTI maps to the z-th filename (natsorted) in the QD PNG directory for
that patient's split. Slice counts must match — both come from the same 1mm
DICOM series.

Usage:
  python export_masks.py \
      --seg_root  segmentations \
      --png_root  /path/to/png_dataset \
      --mask_dir  /path/to/output/masks

  # Re-export only specific patients:
  python export_masks.py ... --patients L014 L058 L196
"""

import argparse
import os
import sys
from glob import glob

import nibabel as nib
import numpy as np
from natsort import natsorted
from PIL import Image


CLASS_MAP = {
    1: "subcutaneous_fat",
    2: "torso_fat",
    3: "skeletal_muscle",
    4: "intermuscular_fat",
}

SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# PNG dataset index
# ---------------------------------------------------------------------------

def build_png_index(png_root):
    """
    Scan png_root/{split}/1mm/QD/*.png and return:
      patient_split[patient_id] = split
      patient_files[patient_id] = natsorted list of filenames (basenames only)
    """
    patient_split = {}
    patient_files = {}
    for split in SPLITS:
        qd_dir = os.path.join(png_root, split, "1mm", "QD")
        if not os.path.isdir(qd_dir):
            continue
        for fpath in natsorted(glob(os.path.join(qd_dir, "*.png"))):
            fname = os.path.basename(fpath)
            patient_id = fname.split("_")[0]   # e.g. "L014" from "L014_QD_1_1_CT_0001.png"
            if patient_id not in patient_files:
                patient_files[patient_id] = []
                patient_split[patient_id] = split
            patient_files[patient_id].append(fname)
    return patient_split, patient_files


# ---------------------------------------------------------------------------
# NIfTI → PNG export
# ---------------------------------------------------------------------------

def export_patient(nii_path, patient_id, split, filenames, mask_dir, overwrite):
    """
    Export one patient's segmentation NIfTI to per-class binary PNG masks.

    The NIfTI is reoriented to RAS canonical (z = inferior→superior) before
    slicing, matching the natural sort order of the QD PNG files.

    Returns the number of slices written, or 0 if skipped.
    """
    vol = nib.load(nii_path)
    vol = nib.as_closest_canonical(vol)     # reorient to RAS: z = inferior→superior
    data = np.asarray(vol.get_fdata(), dtype=np.int32)

    if data.ndim != 3:
        print(f"  [SKIP] {patient_id}: expected 3D volume, got shape {data.shape}")
        return 0

    n_nii = data.shape[2]
    n_png = len(filenames)
    if n_nii != n_png:
        print(
            f"  [SKIP] {patient_id}: NIfTI has {n_nii} slices but PNG dataset "
            f"has {n_png} — slice counts must match. "
            f"Ensure mask_generation.py was run with --dose quarter on the 1mm series."
        )
        return 0

    for class_id, class_name in CLASS_MAP.items():
        out_dir = os.path.join(mask_dir, class_name, split, "1mm", "QD")
        os.makedirs(out_dir, exist_ok=True)
        for z, fname in enumerate(filenames):
            out_path = os.path.join(out_dir, fname)
            if os.path.exists(out_path) and not overwrite:
                continue
            mask = (data[:, :, z] == class_id).astype(np.uint8) * 255
            Image.fromarray(mask, mode="L").save(out_path)

    return n_nii


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Export TotalSegmentator NIfTI segmentations to model-ready PNG masks."
    )
    ap.add_argument(
        "--seg_root", required=True,
        help="Root of the segmentations tree produced by mask_generation.py.",
    )
    ap.add_argument(
        "--png_root", required=True,
        help="Root of the PNG dataset ({split}/1mm/{QD,FD}/*.png).",
    )
    ap.add_argument(
        "--mask_dir", required=True,
        help="Output root for the PNG masks.",
    )
    ap.add_argument(
        "--patients", nargs="*", default=None,
        help="Optional list of patient IDs to process (e.g. L014 L058). "
             "Defaults to all patients found in the PNG dataset.",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Re-export even if the output PNG already exists.",
    )
    args = ap.parse_args()

    # --- build patient index from PNG dataset ---
    print(f"Scanning PNG dataset: {args.png_root}")
    patient_split, patient_files = build_png_index(args.png_root)
    if not patient_files:
        print(
            f"No QD PNG files found under {args.png_root}. "
            "Expected layout: {split}/1mm/QD/*.png",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  {len(patient_files)} patients across splits: "
          + ", ".join(f"{s}={sum(1 for p,sp in patient_split.items() if sp==s)}"
                      for s in SPLITS))

    # --- find all NIfTI files ---
    nii_paths = natsorted(glob(
        os.path.join(args.seg_root, "**", "segmentations.nii.gz"),
        recursive=True,
    ))
    if not nii_paths:
        print(
            f"No segmentations.nii.gz found under {args.seg_root}.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Found {len(nii_paths)} segmentation volume(s) in {args.seg_root}.")

    # --- filter by --patients if given ---
    filter_set = set(args.patients) if args.patients else None

    total_slices = 0
    n_ok = n_skip = n_missing = 0

    for nii_path in nii_paths:
        # Path layout: .../seg_root/{root_name}/{patient_id}/quarter_1mm/segmentations.nii.gz
        parts = nii_path.replace("\\", "/").split("/")
        patient_id = parts[-3]

        if filter_set and patient_id not in filter_set:
            continue

        if patient_id not in patient_split:
            print(f"  [MISSING] {patient_id}: not found in PNG dataset — skipping")
            n_missing += 1
            continue

        if filter_set:
            filter_set.discard(patient_id)   # track which were found

        split = patient_split[patient_id]
        filenames = patient_files[patient_id]

        print(f"  {patient_id:6s} ({split:5s}, {len(filenames)} slices) ← {nii_path}")
        n = export_patient(nii_path, patient_id, split, filenames, args.mask_dir, args.overwrite)
        if n:
            total_slices += n
            n_ok += 1
        else:
            n_skip += 1

    # warn about any requested patients that had no NIfTI
    if filter_set:
        for pid in sorted(filter_set):
            print(f"  [NOT FOUND] {pid}: no segmentations.nii.gz found under seg_root")

    print(f"\nDone. Exported={n_ok}  Skipped={n_skip}  Missing={n_missing}  "
          f"Total slices written: {total_slices}")
    if total_slices > 0:
        print(f"Masks written to: {os.path.abspath(args.mask_dir)}")
        print(f"  Layout: {{class}}/{{split}}/1mm/QD/{{qd_filename}}.png")


if __name__ == "__main__":
    main()
