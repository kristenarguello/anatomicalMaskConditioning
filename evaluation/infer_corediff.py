#!/usr/bin/env python3
"""
CoreDiff inference adapter.

Loads the EMA model checkpoint and runs inference on all test slices,
saving one uint8 prediction PNG per slice.

Variants (match output directory names):
  baseline_no_mask      → corediff_no_mask_ctx   (context=True,  mask=False)
  baseline_no_mask_nc   → corediff_no_mask_noctx (context=False, mask=False)
  masked                → corediff_mask_ctx       (context=True,  mask=True)
  masked_no_context     → corediff_mask_noctx     (context=False, mask=True)

Usage:
  python infer_corediff.py --variant corediff_no_mask_ctx
  python infer_corediff.py --variant corediff_mask_ctx
  python infer_corediff.py --variant corediff_no_mask_noctx
  python infer_corediff.py --variant corediff_mask_noctx
"""

import argparse
import copy
import os
import sys

import numpy as np
from natsort import natsorted
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE         = os.path.dirname(os.path.abspath(__file__))
COREDIFF_ROOT = os.path.join(_HERE, "..", "models", "corediff")
DATASET_ROOT  = os.environ.get("DATASET_ROOT", os.path.join(_HERE, "..", "dataset"))
LDCT_DIR      = os.path.join(DATASET_ROOT, "png_dataset", "test", "1mm", "QD")
MASK_DIR      = os.path.join(DATASET_ROOT, "multilabel")
RESULTS_ROOT  = os.environ.get("RESULTS_ROOT", os.path.join(_HERE, "results"))

TISSUES = ["subcutaneous_fat", "torso_fat", "skeletal_muscle", "intermuscular_fat"]

# Map variant name → (output_dir_name, context, use_mask, in_channels)
# output_dir_name is the actual subdirectory under CoreDiff-segGuided/output/
VARIANTS = {
    "corediff_no_mask_ctx":   ("corediff_baseline_no_mask",         True,  False, 3),
    "corediff_no_mask_noctx": ("corediff_baseline_no_mask_no_context", False, False, 1),
    "corediff_mask_ctx":      ("corediff_masked",                   True,  True,  3),
    "corediff_mask_noctx":    ("corediff_masked_no_context",        False, True,  1),
}

CKPT_ITER = 150000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gray_norm(path: str) -> np.ndarray:
    """Load 8-bit PNG as float32 [0, 1]."""
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def load_mask(qd_name: str) -> np.ndarray:
    """Return (1, H, W) float32 label map normalised to [0, 1]."""
    label_map = np.zeros((512, 512), dtype=np.float32)
    for i, tissue in enumerate(TISSUES):
        path = os.path.join(MASK_DIR, tissue, "test", "1mm", "QD", qd_name)
        if os.path.exists(path):
            m = np.array(Image.open(path).convert("L"))
            label_map[m > 0] = float(i + 1)
    return (label_map / 4.0)[np.newaxis]     # (1, H, W)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(variant: str, device_str: str = ""):
    if variant not in VARIANTS:
        print(f"Unknown variant '{variant}'. Choose from: {list(VARIANTS)}")
        sys.exit(1)

    output_dir_name, context, use_mask, in_channels = VARIANTS[variant]

    out_dir = os.path.join(RESULTS_ROOT, variant, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    ema_path = os.path.join(
        COREDIFF_ROOT, "output", output_dir_name, "save_models",
        f"ema_model-{CKPT_ITER}"
    )
    if not os.path.exists(ema_path):
        print(f"[ERROR] checkpoint not found: {ema_path}")
        sys.exit(1)

    sys.path.insert(0, COREDIFF_ROOT)
    from models.corediff.corediff_wrapper import Network       # noqa
    from models.corediff.diffusion_modules import Diffusion    # noqa

    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    denoise_fn = Network(
        in_channels=in_channels,
        out_channels=1,
        context=context,
        use_masks=use_mask,
    )
    ema_model = Diffusion(
        denoise_fn=denoise_fn,
        image_size=512,
        timesteps=10,
        context=context,
    ).to(device)

    state = torch.load(ema_path, map_location=device)
    # strip DataParallel prefix if present
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    ema_model.load_state_dict(state)
    ema_model.eval()

    ldct_files = natsorted(f for f in os.listdir(LDCT_DIR) if f.endswith(".png"))
    print(
        f"[infer_corediff] variant={variant}  slices={len(ldct_files)}"
        f"  context={context}  use_mask={use_mask}  device={device}"
    )

    # Context: skip first and last slice (no neighbours)
    if context:
        valid_range = range(1, len(ldct_files) - 1)
    else:
        valid_range = range(len(ldct_files))

    with torch.no_grad():
        for idx in valid_range:
            qd_name = ldct_files[idx]
            out_path = os.path.join(out_dir, qd_name)
            if os.path.exists(out_path):
                continue

            if context:
                prev_slice = load_gray_norm(os.path.join(LDCT_DIR, ldct_files[idx - 1]))
                curr_slice = load_gray_norm(os.path.join(LDCT_DIR, ldct_files[idx]))
                next_slice = load_gray_norm(os.path.join(LDCT_DIR, ldct_files[idx + 1]))
                inp = np.stack([prev_slice, curr_slice, next_slice], axis=0)   # (3,H,W)
            else:
                curr_slice = load_gray_norm(os.path.join(LDCT_DIR, ldct_files[idx]))
                inp = curr_slice[np.newaxis]    # (1,H,W)

            x = torch.from_numpy(inp).unsqueeze(0).float().to(device)   # (1,C,H,W)

            mask_tensor = None
            if use_mask:
                mask_np = load_mask(qd_name)          # (1,H,W)
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float().to(device)

            gen, _, _ = ema_model.sample(
                batch_size=1,
                img=x,
                t=10,
                sampling_routine="ddim",
                n_iter=CKPT_ITER,
                start_adjust_iter=1,
                mask=mask_tensor,
            )

            pred_np = gen.squeeze().cpu().numpy()       # [0, 1] (clamped in sample())
            Image.fromarray(to_uint8(pred_np), mode="L").save(out_path)

            if (idx + 1) % 200 == 0:
                print(f"  {idx+1}/{len(ldct_files)}", end="\r", flush=True)

    print(f"\n[infer_corediff] done → {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        choices=list(VARIANTS),
        required=True,
        help="corediff_no_mask_ctx | corediff_no_mask_noctx | "
             "corediff_mask_ctx | corediff_mask_noctx",
    )
    p.add_argument("--device", default="", help="e.g. cuda:0")
    args = p.parse_args()
    run(args.variant, args.device)
