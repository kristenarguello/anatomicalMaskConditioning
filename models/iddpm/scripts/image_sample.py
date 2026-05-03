"""
SDEdit-style LDCT -> NDCT translation sampling.

For each LDCT image the script:
  1. Adds noise up to timestep t0 = round(trans_noise_level * T)
  2. Runs the reverse diffusion from t0 down to 0 with LDCT (and optionally
     the segmentation mask) concatenated at every denoising step.

CT input layout
---------------
  --ldct_dir  path/to/{split}/1mm/QD/

Mask input layout (only when --use_mask True)
----------------------------------------------
  --mask_root  path/to/masks_root

  The script constructs mask paths as:
    {mask_root}/{class_name}/{split}/1mm/QD/{filename}.png

  where {split} = basename(ldct_dir's grandparent) and class_name is one of:
    subcutaneous_fat  torso_fat  skeletal_muscle  intermuscular_fat

  The four binary masks are merged into a single multi-class label map
  (overlay: label 1-4 per class, background=0) normalised to [0,1].

Output
------
  One uint16 PNG per input slice saved to --out_dir as <stem>_pred.png.
"""

import argparse
import os
from pathlib import Path

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import MASK_CLASS_NAMES
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_ct_image(path, resolution):
    """Load a single-channel CT PNG and normalise to [-1, 1]."""
    with bf.BlobFile(str(path), "rb") as f:
        img = Image.open(f)
        img.load()
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), resample=Image.BICUBIC)
    arr = np.array(img)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 32767.5 - 1.0
    else:
        arr = arr.astype(np.float32) / 127.5 - 1.0
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    else:
        arr = arr[:1]
    return arr


def load_combined_mask(ldct_path, mask_class_dirs, resolution):
    """
    Load the 4 binary mask PNGs for a single slice and merge via overlay.

    Returns float32 [1, H, W] with values in [0, 1].
    """
    filename = os.path.basename(str(ldct_path))
    combined = None
    for class_idx, mask_dir in enumerate(mask_class_dirs):
        mask_path = bf.join(mask_dir, filename)
        with bf.BlobFile(mask_path, "rb") as f:
            img = Image.open(f)
            img.load()
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), resample=Image.NEAREST)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if combined is None:
            combined = np.zeros(arr.shape, dtype=np.float32)
        combined[arr > 0] = float(class_idx + 1)

    combined = combined / 4.0   # normalise to [0, 1]
    return combined[np.newaxis]  # [1, H, W]


def save_uint16_png(arr_1hw, path):
    """Save a [1, H, W] float32 array in [-1, 1] as a 16-bit grayscale PNG."""
    arr = ((arr_1hw[0] + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)
    Image.fromarray(arr, mode="I;16").save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # Collect LDCT files
    ldct_files = sorted(
        p for p in Path(args.ldct_dir).iterdir() if p.suffix.lower() == ".png"
    )
    assert len(ldct_files) > 0, f"No PNG files found in {args.ldct_dir}"

    # Build the 4 mask class directories if needed
    mask_class_dirs = None
    if args.use_mask:
        assert args.mask_root, "--mask_root must be set when --use_mask True"
        # Derive split from ldct_dir: .../split/1mm/QD -> split
        split = Path(args.ldct_dir).parents[1].name
        mask_class_dirs = [
            bf.join(args.mask_root, cls, split, "1mm", "QD")
            for cls in MASK_CLASS_NAMES
        ]
        logger.log(f"mask split = '{split}', classes = {MASK_CLASS_NAMES}")

    os.makedirs(args.out_dir, exist_ok=True)

    # t0 in discrete timestep units
    t0 = int(args.trans_noise_level * diffusion.num_timesteps)
    logger.log(
        f"SDEdit t0 = {t0} / {diffusion.num_timesteps}  "
        f"(trans_noise_level = {args.trans_noise_level})"
    )

    device = dist_util.dev()
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    # Skip already-processed slices so runs can be resumed
    ldct_files = [
        p for p in ldct_files
        if not os.path.exists(os.path.join(args.out_dir, f"{p.stem}_pred.png"))
    ]
    # Partition across GPUs when world_size > 1
    if args.world_size > 1:
        ldct_files = ldct_files[args.rank::args.world_size]
        logger.log(f"rank={args.rank}/{args.world_size}: {len(ldct_files)} slices assigned")
    else:
        logger.log(f"{len(ldct_files)} slices left to process")

    idx = 0
    while idx < len(ldct_files):
        batch_files = ldct_files[idx : idx + args.batch_size]
        B = len(batch_files)

        # ---- Load batch ------------------------------------------------
        ldct_np = np.stack([
            load_ct_image(p, args.image_size) for p in batch_files
        ])  # [B, 1, H, W]
        ldct_tensor = th.from_numpy(ldct_np).to(device)

        model_kwargs = {"ldct": ldct_tensor}
        if mask_class_dirs is not None:
            mask_np = np.stack([
                load_combined_mask(p, mask_class_dirs, args.image_size)
                for p in batch_files
            ])  # [B, 1, H, W]
            model_kwargs["mask"] = th.from_numpy(mask_np).to(device)

        # ---- SDEdit: x_t0 = sqrt(abar_t0)*ldct + sqrt(1-abar_t0)*noise
        t0_batch = th.tensor([t0] * B, device=device)
        x_t0 = diffusion.q_sample(ldct_tensor, t=t0_batch)  # [B, 1, H, W]

        # ---- Reverse diffusion from t0 -> 0 ----------------------------
        with th.no_grad():
            samples = sample_fn(
                model,
                (B, 1, args.image_size, args.image_size),
                noise=x_t0,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True,
                t_start=t0,
            )  # [B, 1, H, W] in [-1, 1]

        # ---- Save outputs -----------------------------------------------
        for i in range(B):
            stem = batch_files[i].stem
            out_path = os.path.join(args.out_dir, f"{stem}_pred.png")
            save_uint16_png(samples[i].cpu().numpy(), out_path)

        logger.log(
            f"saved {min(idx + B, len(ldct_files))} / {len(ldct_files)} slices"
        )
        idx += B

    dist.barrier()
    logger.log(f"done — outputs written to {args.out_dir}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def create_argparser():
    defaults = dict(
        # Checkpoint
        model_path="",
        # Input paths
        ldct_dir="",
        mask_root="",   # root of binary mask dataset (needed when --use_mask True)
        # Output
        out_dir="samples_out",
        # SDEdit
        trans_noise_level=0.4,
        # Sampling
        clip_denoised=True,
        batch_size=4,
        use_ddim=False,
        rank=0,
        world_size=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
