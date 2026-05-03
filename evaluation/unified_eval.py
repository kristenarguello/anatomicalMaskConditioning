#!/usr/bin/env python3
"""
Unified evaluation script for CT denoising models.

Loads model predictions (uint8 PNG, same filename as LDCT) and computes:
  - PSNR, SSIM, RMSE_HU, CNR
  - Boundary vs interior regional PSNR/SSIM
  - Saves per_slice_metrics.csv
  - Saves qualitative PNG images for FIXED_SLICES

Convention (matching RED-CNN's test_metrics.csv):
  images in float [0, 255]; soft-tissue truncation clips > TRUNC_MAX;
  data_range = TRUNC_MAX - TRUNC_MIN = 400.

Usage:
  python unified_eval.py \\
      --pred_dir  results/redcnn_no_mask/predictions \\
      --out_dir   results/redcnn_no_mask \\
      --model_name redcnn --variant no_mask

  # Skip full CSV, only write qualitative images for FIXED_SLICES:
  python unified_eval.py ... --fixed_only
"""

import argparse
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.join(_HERE, "..", "dataset"))
LDCT_DIR     = os.path.join(DATASET_ROOT, "png_dataset", "test", "1mm", "QD")
NDCT_DIR     = os.path.join(DATASET_ROOT, "png_dataset", "test", "1mm", "FD")
MASK_DIR     = os.path.join(DATASET_ROOT, "multilabel")

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(_HERE, "results"))

TISSUES = ["subcutaneous_fat", "torso_fat", "skeletal_muscle", "intermuscular_fat"]

TRUNC_MIN = -160.0
TRUNC_MAX = 240.0
DATA_RANGE = TRUNC_MAX - TRUNC_MIN  # 400.0

# 10 fixed slice indices (0-based, from the sorted test filelist)
FIXED_SLICE_INDICES = [0, 190, 380, 570, 760, 950, 1140, 1330, 1520, 1710]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def sorted_ldct_files():
    files = sorted(f for f in os.listdir(LDCT_DIR) if f.endswith(".png"))
    return files


def ndct_name(qd_name: str) -> str:
    return qd_name.replace("_QD_", "_FD_")


def load_uint8(path: str) -> np.ndarray:
    """Load grayscale PNG; handles uint8 and uint16 (decodes to [0, 255])."""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 256:          # uint16 from iDDPM
        arr = arr / 257.0        # max(uint16)=65535 → 255
    return arr                   # float32 [0, 255]


def load_label_map(qd_name: str) -> np.ndarray:
    """Return integer label map (0–4), shape (H, W)."""
    label_map = np.zeros((512, 512), dtype=np.int32)
    for i, tissue in enumerate(TISSUES):
        path = os.path.join(MASK_DIR, tissue, "test", "1mm", "QD", qd_name)
        if os.path.exists(path):
            mask = np.array(Image.open(path).convert("L"))
            label_map[mask > 0] = i + 1
    return label_map


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def trunc(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, TRUNC_MIN, TRUNC_MAX)


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(DATA_RANGE ** 2 / mse)


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """Returns (ssim_score, ssim_map)."""
    score, ssim_map = structural_similarity(
        pred, gt, data_range=DATA_RANGE, full=True
    )
    return float(score), ssim_map


def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def compute_cnr(img: np.ndarray, label_map: np.ndarray) -> float:
    roi = label_map > 0
    bg = label_map == 0
    if roi.sum() == 0 or bg.sum() == 0:
        return 0.0
    bg_std = float(img[bg].std())
    if bg_std == 0:
        return 0.0
    return float(abs(img[roi].mean() - img[bg].mean()) / bg_std)


def get_boundary_mask(label_map: np.ndarray, width: int = 5) -> np.ndarray:
    """Boolean boundary mask: dilation XOR erosion for each non-BG label."""
    kernel = np.ones((3, 3), np.uint8)
    boundary = np.zeros(label_map.shape, dtype=bool)
    for label in range(1, 5):
        binary = (label_map == label).astype(np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=width // 2)
        eroded = cv2.erode(binary, kernel, iterations=width // 2)
        boundary |= (dilated - eroded).astype(bool)
    return boundary


def regional_psnr(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> float:
    if not region.any():
        return float("nan")
    mse = float(np.mean((pred[region] - gt[region]) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(DATA_RANGE ** 2 / mse)


def compute_regional_metrics(
    pred: np.ndarray, gt: np.ndarray, label_map: np.ndarray
) -> dict:
    boundary = get_boundary_mask(label_map)
    interior = (~boundary) & (label_map > 0)

    _, ssim_map = structural_similarity(
        pred, gt, data_range=DATA_RANGE, full=True
    )

    return {
        "boundary_psnr": regional_psnr(pred, gt, boundary),
        "boundary_ssim": float(ssim_map[boundary].mean()) if boundary.any() else float("nan"),
        "interior_psnr": regional_psnr(pred, gt, interior),
        "interior_ssim": float(ssim_map[interior].mean()) if interior.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# Qualitative output
# ---------------------------------------------------------------------------

QUAL_ROOT = os.path.join(RESULTS_ROOT, "qualitative")


def save_qualitative(
    qd_name: str,
    ldct: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    model_name: str,
    variant: str,
):
    """Save per-slice prediction and side-by-side comparison."""
    slice_dir = os.path.join(QUAL_ROOT, os.path.splitext(qd_name)[0])
    os.makedirs(slice_dir, exist_ok=True)

    def _save_u8(arr, fname):
        clipped = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(clipped, mode="L").save(os.path.join(slice_dir, fname))

    _save_u8(ldct, "ldct.png")
    _save_u8(gt, "ndct_gt.png")
    _save_u8(pred, f"{model_name}_{variant}_pred.png")

    # Side-by-side comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = 0, 255
    for ax, img, title in zip(
        axes,
        [ldct, pred, gt],
        ["LDCT (input)", f"{model_name} {variant}", "NDCT (GT)"],
    ):
        ax.imshow(np.clip(img, vmin, vmax), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(
        os.path.join(slice_dir, f"compare_{model_name}_{variant}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    pred_dir: str,
    out_dir: str,
    model_name: str,
    variant: str,
    fixed_only: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    all_slices = sorted_ldct_files()

    if not os.path.isdir(pred_dir):
        print(f"[ERROR] prediction dir not found: {pred_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    missing = 0

    for idx, qd_name in enumerate(all_slices):
        pred_path = os.path.join(pred_dir, qd_name)
        if not os.path.exists(pred_path):
            missing += 1
            continue

        ldct = load_uint8(os.path.join(LDCT_DIR, qd_name))
        gt = load_uint8(os.path.join(NDCT_DIR, ndct_name(qd_name)))
        pred = load_uint8(pred_path)
        label_map = load_label_map(qd_name)

        # Apply soft-tissue truncation (matching RED-CNN convention)
        pred_t = trunc(pred)
        gt_t = trunc(gt)

        if fixed_only and idx not in FIXED_SLICE_INDICES:
            # Still need SSIM map for regional — skip full loop if fixed_only
            continue

        psnr_val = compute_psnr(pred_t, gt_t)
        ssim_val, _ = compute_ssim(pred_t, gt_t)
        rmse_val = compute_rmse(pred_t, gt_t)
        cnr_val = compute_cnr(pred_t, label_map)
        regional = compute_regional_metrics(pred_t, gt_t, label_map)

        rows.append(
            {
                "slice_name": qd_name,
                "psnr": round(psnr_val, 6),
                "ssim": round(ssim_val, 6),
                "rmse_hu": round(rmse_val, 6),
                "cnr": round(cnr_val, 6),
                "boundary_psnr": round(regional["boundary_psnr"], 6),
                "boundary_ssim": round(regional["boundary_ssim"], 6),
                "interior_psnr": round(regional["interior_psnr"], 6),
                "interior_ssim": round(regional["interior_ssim"], 6),
            }
        )

        if idx in FIXED_SLICE_INDICES:
            save_qualitative(qd_name, ldct, gt, pred, model_name, variant)

        if (len(rows)) % 100 == 0:
            print(f"  processed {len(rows)} slices...", end="\r", flush=True)

    print()

    if missing:
        print(f"[WARN] {missing} slices missing from pred_dir (skipped).")

    # Write CSV
    fieldnames = [
        "slice_name", "psnr", "ssim", "rmse_hu", "cnr",
        "boundary_psnr", "boundary_ssim", "interior_psnr", "interior_ssim",
    ]
    csv_path = os.path.join(out_dir, "per_slice_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    if rows:
        for metric in ["psnr", "ssim", "rmse_hu", "cnr"]:
            vals = [r[metric] for r in rows if not np.isnan(r[metric])]
            print(
                f"  {metric:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}"
            )

    print(f"\nSaved {len(rows)} rows → {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unified CT denoising evaluation")
    p.add_argument("--pred_dir", required=True,
                   help="Directory of prediction PNGs (named as LDCT files)")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for CSV and figures")
    p.add_argument("--model_name", required=True)
    p.add_argument("--variant", required=True,
                   help="e.g. no_mask / mask")
    p.add_argument("--fixed_only", action="store_true",
                   help="Only process FIXED_SLICE_INDICES (for qualitative saves)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"[unified_eval] {args.model_name}/{args.variant}")
    print(f"  pred_dir : {args.pred_dir}")
    print(f"  out_dir  : {args.out_dir}")
    evaluate(
        pred_dir=args.pred_dir,
        out_dir=args.out_dir,
        model_name=args.model_name,
        variant=args.variant,
        fixed_only=args.fixed_only,
    )
