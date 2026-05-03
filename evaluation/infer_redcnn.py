#!/usr/bin/env python3
"""
RED-CNN inference adapter.

Loads the best checkpoint and runs inference on all test slices, saving
one uint8 prediction PNG per slice (named identically to the LDCT input).

Usage:
  python infer_redcnn.py --variant no_mask
  python infer_redcnn.py --variant mask
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE        = os.path.dirname(os.path.abspath(__file__))
REDCNN_ROOT  = os.path.join(_HERE, "..", "models", "red-cnn")
DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.join(_HERE, "..", "dataset"))
LDCT_DIR     = os.path.join(DATASET_ROOT, "png_dataset", "test", "1mm", "QD")
MASK_DIR     = os.path.join(DATASET_ROOT, "multilabel")
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(_HERE, "results"))

TISSUES = ["subcutaneous_fat", "torso_fat", "skeletal_muscle", "intermuscular_fat"]

NORM_MIN, NORM_MAX = -1024.0, 3072.0

VARIANT_TO_SAVE = {
    "no_mask": os.path.join(REDCNN_ROOT, "save", "baseline"),
    "mask": os.path.join(REDCNN_ROOT, "save", "masked_zeroinit"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_png_normalized(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    arr = np.clip(arr, NORM_MIN, NORM_MAX)
    arr = (arr - NORM_MIN) / (NORM_MAX - NORM_MIN)
    return arr


def load_mask(qd_name: str) -> np.ndarray:
    label_map = np.zeros((512, 512), dtype=np.float32)
    for i, tissue in enumerate(TISSUES):
        path = os.path.join(MASK_DIR, tissue, "test", "1mm", "QD", qd_name)
        if os.path.exists(path):
            m = np.array(Image.open(path).convert("L"), dtype=np.float32)
            label_map[m > 0] = float(i + 1)
    return label_map / 4.0


def denormalize(arr: np.ndarray) -> np.ndarray:
    return arr * (NORM_MAX - NORM_MIN) + NORM_MIN


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(variant: str, device_str: str = "", batch_size: int = 32):
    save_path = VARIANT_TO_SAVE[variant]
    out_dir = os.path.join(RESULTS_ROOT, f"redcnn_{variant}", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, REDCNN_ROOT)
    from networks import RED_CNN  # noqa: E402

    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    use_mask = (variant == "mask")
    in_channels = 2 if use_mask else 1
    model = RED_CNN(in_channels=in_channels).to(device)

    ckpt_path = os.path.join(save_path, "best.ckpt")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # strip DataParallel prefix if present
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    ldct_files = sorted(f for f in os.listdir(LDCT_DIR) if f.endswith(".png"))
    # Filter already-done slices
    todo = [f for f in ldct_files if not os.path.exists(os.path.join(out_dir, f))]
    print(
        f"[infer_redcnn] variant={variant}  total={len(ldct_files)}"
        f"  todo={len(todo)}  batch={batch_size}  device={device}"
    )

    with torch.no_grad():
        for start in range(0, len(todo), batch_size):
            batch_names = todo[start : start + batch_size]

            ldcts = np.stack([
                load_png_normalized(os.path.join(LDCT_DIR, n)) for n in batch_names
            ])  # (B, H, W)
            x = torch.from_numpy(ldcts).unsqueeze(1).float().to(device)  # (B,1,H,W)

            if use_mask:
                masks = np.stack([load_mask(n) for n in batch_names])    # (B, H, W)
                m = torch.from_numpy(masks).unsqueeze(1).float().to(device)
                inp = torch.cat([x, m], dim=1)
            else:
                inp = x

            preds = model(inp)                                    # (B,1,H,W)
            preds_np = preds.squeeze(1).cpu().numpy()             # (B,H,W)

            for pred_np, qd_name in zip(preds_np, batch_names):
                pred_hu = denormalize(pred_np)
                out_path = os.path.join(out_dir, qd_name)
                Image.fromarray(to_uint8(pred_hu), mode="L").save(out_path)

            done = min(start + batch_size, len(todo))
            print(f"  {done}/{len(todo)}", end="\r", flush=True)

    print(f"\n[infer_redcnn] done → {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["no_mask", "mask"], required=True)
    p.add_argument("--device", default="", help="e.g. cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    run(args.variant, args.device, args.batch_size)
