#!/usr/bin/env python3
"""
SegGuidedDiff (seg-guided-diff) inference adapter.

Loads a trained DDPM from the HuggingFace-style checkpoint directory and runs
SDEdit image-translation inference on all test slices.

Variants:
  no_mask   – ddpm-mayo_challenge-512          (no segmentation guidance)
  mask      – ddpm-mayo_challenge-512-segguided (segmentation-guided)

Usage:
  python infer_segguideddiff.py --variant no_mask
  python infer_segguideddiff.py --variant mask
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from natsort import natsorted
from PIL import Image
import torch
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE        = os.path.dirname(os.path.abspath(__file__))
SGD_ROOT     = os.path.join(_HERE, "..", "models", "seg-guided-diff")
DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.join(_HERE, "..", "dataset"))
LDCT_DIR     = os.path.join(DATASET_ROOT, "png_dataset", "test", "1mm", "QD")
NDCT_DIR     = os.path.join(DATASET_ROOT, "png_dataset", "test", "1mm", "FD")
MASK_ROOT    = os.path.join(DATASET_ROOT, "multilabel")
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(_HERE, "results"))

TISSUES = ["subcutaneous_fat", "torso_fat", "skeletal_muscle", "intermuscular_fat"]

VARIANTS = {
    "no_mask": {
        "run_dir": os.path.join(SGD_ROOT, "ddpm-mayo_challenge-512"),
        "segmentation_guided": False,
    },
    "mask": {
        "run_dir": os.path.join(SGD_ROOT, "ddpm-mayo_challenge-512-segguided"),
        "segmentation_guided": True,
    },
}

TRANS_NOISE_LEVEL = 0.2      # must match training config
NUM_INFERENCE_STEPS = 1000
BATCH_SIZE = 16
IMAGE_SIZE = 512


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

# Images: normalize to [-1, 1] (matches training preprocessing)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Masks: ToTensor only → [0, 1], NEAREST to preserve binary values
mask_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


def load_pil(path: str) -> torch.Tensor:
    """Load PNG as 1-channel tensor in [-1, 1]."""
    img = Image.open(path).convert("L")
    return preprocess(img)       # (1, H, W) in [-1, 1]


def load_seg_batch(qd_names: list, device: torch.device) -> dict:
    """Build a seg_batch dict (no seg guidance: only lds_cond + images)."""
    ldct_batch, ndct_batch = [], []
    for qd_name in qd_names:
        ldct_batch.append(load_pil(os.path.join(LDCT_DIR, qd_name)))
        ndct_name = qd_name.replace("_QD_", "_FD_")
        ndct_batch.append(load_pil(os.path.join(NDCT_DIR, ndct_name)))
    return {
        "lds_cond": torch.stack(ldct_batch).to(device),
        "images":   torch.stack(ndct_batch).to(device),
        "image_filenames": list(qd_names),
    }


def load_seg_batch_guided(qd_names: list, device: torch.device) -> dict:
    """Build a seg_batch with segmentation channels."""
    batch = load_seg_batch(qd_names, device)
    for tissue in TISSUES:
        masks = []
        for qd_name in qd_names:
            mask_path = os.path.join(MASK_ROOT, tissue, "test", "1mm", "QD", qd_name)
            if os.path.exists(mask_path):
                masks.append(mask_preprocess(Image.open(mask_path).convert("L")))
            else:
                masks.append(torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE))
        batch[f"seg_{tissue}"] = torch.stack(masks).to(device)
    return batch


def to_uint8(tensor_0_1: np.ndarray) -> np.ndarray:
    return np.clip(tensor_0_1 * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run(variant: str, device_str: str = "", rank: int = 0, world_size: int = 1):
    cfg = VARIANTS[variant]
    run_dir = cfg["run_dir"]
    seg_guided = cfg["segmentation_guided"]

    out_dir = os.path.join(RESULTS_ROOT, f"segguideddiff_{variant}", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, SGD_ROOT)

    import diffusers
    from diffusers import DDPMScheduler
    from eval import SegGuidedDDPMPipeline, LDConditionedDDPMPipeline  # noqa
    from training import TrainingConfig                                   # noqa

    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    unet = diffusers.UNet2DModel.from_pretrained(
        os.path.join(run_dir, "unet")
    ).to(device)
    unet.eval()

    scheduler = DDPMScheduler.from_pretrained(
        os.path.join(run_dir, "scheduler")
    )

    config = TrainingConfig(
        model_type="DDPM",
        image_size=IMAGE_SIZE,
        dataset="mayo_challenge",
        segmentation_guided=seg_guided,
        segmentation_channel_mode="single",
        num_segmentation_classes=5 if seg_guided else None,
        eval_batch_size=BATCH_SIZE,
        trans_noise_level=TRANS_NOISE_LEVEL,
        use_ablated_segmentations=False,
        class_conditional=False,
        use_cfg_for_eval_conditioning=False,
        output_dir=run_dir,
    )

    if seg_guided:
        pipeline = SegGuidedDDPMPipeline(
            unet=unet,
            scheduler=scheduler,
            eval_dataloader=None,       # not used in __call__
            external_config=config,
        )
    else:
        pipeline = LDConditionedDDPMPipeline(
            unet=unet,
            scheduler=scheduler,
            external_config=config,
        )

    ldct_files = natsorted(f for f in os.listdir(LDCT_DIR) if f.endswith(".png"))
    if world_size > 1:
        ldct_files = ldct_files[rank::world_size]
    print(
        f"[infer_segguideddiff] variant={variant}  slices={len(ldct_files)}"
        f"  seg_guided={seg_guided}  device={device}"
        + (f"  rank={rank}/{world_size}" if world_size > 1 else "")
    )

    # Process in batches
    i = 0
    while i < len(ldct_files):
        batch_names = []
        for j in range(i, min(i + BATCH_SIZE, len(ldct_files))):
            if not os.path.exists(os.path.join(out_dir, ldct_files[j])):
                batch_names.append(ldct_files[j])

        if batch_names:
            with torch.no_grad():
                if seg_guided:
                    seg_batch = load_seg_batch_guided(batch_names, device)
                    out = pipeline(
                        batch_size=len(batch_names),
                        seg_batch=seg_batch,
                        translate=True,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        output_type="np",
                    )
                else:
                    seg_batch = load_seg_batch(batch_names, device)
                    out = pipeline(
                        batch_size=len(batch_names),
                        seg_batch=seg_batch,
                        translate=True,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        output_type="np",
                    )

            # out.images: (B, H, W, 1) in [0, 1]
            for k, qd_name in enumerate(batch_names):
                pred_arr = out.images[k, :, :, 0]   # (H, W) in [0, 1]
                Image.fromarray(to_uint8(pred_arr), mode="L").save(
                    os.path.join(out_dir, qd_name)
                )

        i += BATCH_SIZE
        if i % 200 == 0:
            print(f"  {i}/{len(ldct_files)}", end="\r", flush=True)

    print(f"\n[infer_segguideddiff] done → {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["no_mask", "mask"], required=True)
    p.add_argument("--gpus", default="0",
                   help="Comma-separated GPU IDs, e.g. '3' or '3,0' for two GPUs")
    p.add_argument("--rank", type=int, default=-1,
                   help="Worker rank (set automatically when forked by multi-GPU mode)")
    args = p.parse_args()

    gpu_list = [g.strip().replace("cuda:", "") for g in args.gpus.split(",") if g.strip()]
    world_size = len(gpu_list)

    if world_size > 1 and args.rank == -1:
        # Top-level call: fork one worker subprocess per GPU
        procs = []
        for rank, gpu in enumerate(gpu_list):
            cmd = [
                sys.executable, __file__,
                "--variant", args.variant,
                "--gpus", gpu,
                "--rank", str(rank),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            procs.append(subprocess.Popen(cmd, env=env))
        for proc in procs:
            ret = proc.wait()
            if ret != 0:
                sys.exit(ret)
    else:
        # Single-GPU or worker subprocess.
        # When running as a worker, CUDA_VISIBLE_DEVICES is already set by the
        # parent to the correct physical GPU, so the device is always cuda:0.
        if args.rank >= 0:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            gpu = gpu_list[0] if gpu_list else ""
            device_str = f"cuda:{gpu}" if gpu else ""
        rank = max(args.rank, 0)
        run(args.variant, device_str, rank=rank, world_size=max(world_size, 1) if args.rank >= 0 else 1)
