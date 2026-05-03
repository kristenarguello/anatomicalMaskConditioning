"""
Train a diffusion model on images.

For CT image translation (LDCT -> NDCT), set --image_channels 1 and point
--data_dir at a directory with the layout:

    <data_dir>/1mm/QD/*.png         (low-dose CT)
    <data_dir>/1mm/FD/*.png         (full-dose CT, training target)

Pass --use_mask True and --mask_root to enable the 3-channel variant.

Optional periodic validation
-----------------------------
Set --val_data_dir (same layout as data_dir but for the val split) and
--val_interval N to compute PSNR / SSIM / RMSE every N training steps on a
small fixed subset of val slices (--val_num_samples, default 16).
Metrics are written to the same logger as training losses.
"""

import argparse
import csv
import os
from pathlib import Path

import blobfile as bf
import numpy as np
import torch as th

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data, load_ct_data, MASK_CLASS_NAMES
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _load_ct_arr(path, resolution):
    """Load a single-channel CT PNG -> float32 [1, H, W] in [-1, 1]."""
    from PIL import Image
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
    return arr[np.newaxis] if arr.ndim == 2 else arr[:1]


def _load_combined_mask_arr(ldct_path, mask_class_dirs, resolution):
    """Load 4 binary masks and merge via overlay -> float32 [1, H, W] in [0, 1]."""
    from PIL import Image
    filename = os.path.basename(str(ldct_path))
    combined = None
    for class_idx, d in enumerate(mask_class_dirs):
        with bf.BlobFile(bf.join(d, filename), "rb") as f:
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
    return (combined / 4.0)[np.newaxis]


def make_val_fn(args, diffusion, device):
    """
    Pre-load val images and return a val_fn(model, step) closure.
    Returns None if --val_data_dir is not set.
    """
    if not args.val_data_dir:
        return None

    res = args.image_size
    ldct_dir = os.path.join(args.val_data_dir, "1mm", "QD")
    ndct_dir = os.path.join(args.val_data_dir, "1mm", "FD")

    ldct_files = sorted(Path(ldct_dir).glob("*.png"))[: args.val_num_samples]
    ndct_files = sorted(Path(ndct_dir).glob("*.png"))[: args.val_num_samples]
    assert len(ldct_files) > 0, f"No PNGs found in {ldct_dir}"
    assert len(ldct_files) == len(ndct_files), "Val LDCT/NDCT count mismatch"

    logger.log(f"Loading {len(ldct_files)} val slices for periodic evaluation…")

    ldct_np = np.stack([_load_ct_arr(p, res) for p in ldct_files])   # [N,1,H,W]
    ndct_np = np.stack([_load_ct_arr(p, res) for p in ndct_files])   # [N,1,H,W]

    mask_class_dirs = None
    mask_np = None
    if args.use_mask:
        split = os.path.basename(args.val_data_dir.rstrip("/"))
        mask_class_dirs = [
            bf.join(args.mask_root, cls, split, "1mm", "QD")
            for cls in MASK_CLASS_NAMES
        ]
        mask_np = np.stack([
            _load_combined_mask_arr(p, mask_class_dirs, res) for p in ldct_files
        ])  # [N,1,H,W]

    t0 = int(args.trans_noise_level * diffusion.num_timesteps)

    def val_fn(model, step):
        was_training = model.training
        model.eval()

        ldct = th.from_numpy(ldct_np).to(device)
        ndct_gt = ndct_np  # kept on CPU for metric computation

        model_kwargs = {"ldct": ldct}
        if mask_np is not None:
            model_kwargs["mask"] = th.from_numpy(mask_np).to(device)

        t0_batch = th.tensor([t0] * len(ldct), device=device)

        with th.no_grad():
            x_t0 = diffusion.q_sample(ldct, t=t0_batch)
            samples = diffusion.p_sample_loop(
                model,
                (len(ldct), 1, res, res),
                noise=x_t0,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                t_start=t0,
            )  # [N, 1, H, W]

        pred = samples.cpu().numpy()  # [N, 1, H, W]

        psnr_vals, ssim_vals, rmse_vals = [], [], []
        for i in range(len(pred)):
            p = pred[i, 0]      # [H, W]
            g = ndct_gt[i, 0]   # [H, W]
            mse = float(np.mean((p - g) ** 2))
            rmse_vals.append(np.sqrt(mse))
            # PSNR over [-1,1] range (data_range = 2)
            psnr_vals.append(20.0 * np.log10(2.0 / max(np.sqrt(mse), 1e-8)))
            try:
                from skimage.metrics import structural_similarity
                ssim_vals.append(
                    structural_similarity(g, p, data_range=2.0)
                )
            except ImportError:
                pass

        mean_psnr = float(np.mean(psnr_vals))
        mean_rmse = float(np.mean(rmse_vals))
        mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else None

        # --- main training log (alongside loss curves) ---
        logger.logkv("val_psnr", mean_psnr)
        logger.logkv("val_rmse", mean_rmse)
        if mean_ssim is not None:
            logger.logkv("val_ssim", mean_ssim)
        logger.dumpkvs()

        # --- dedicated CSV so metrics are easy to plot/inspect ---
        # Only rank 0 writes to avoid concurrent writes in multi-GPU runs.
        import torch.distributed as _dist
        log_dir = logger.get_dir()
        is_rank0 = not _dist.is_initialized() or _dist.get_rank() == 0
        if log_dir and is_rank0:
            csv_path = os.path.join(log_dir, "val_metrics.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                fieldnames = ["step", "psnr", "rmse"] + (["ssim"] if mean_ssim is not None else [])
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                row = {"step": step, "psnr": mean_psnr, "rmse": mean_rmse}
                if mean_ssim is not None:
                    row["ssim"] = mean_ssim
                writer.writerow(row)
            logger.log(f"[val] step={step}  PSNR={mean_psnr:.3f}  RMSE={mean_rmse:.5f}"
                       + (f"  SSIM={mean_ssim:.4f}" if mean_ssim is not None else ""))

        if was_training:
            model.train()

    return val_fn


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
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.image_channels != 3:
        data = load_ct_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            use_mask=args.use_mask,
            mask_root=args.mask_root,
        )
    else:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )

    val_fn = make_val_fn(args, diffusion, dist_util.dev())
    if val_fn is not None:
        logger.log(
            f"Validation enabled: every {args.val_interval} steps "
            f"on {args.val_num_samples} slices (t0={int(args.trans_noise_level * diffusion.num_timesteps)})"
        )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        val_fn=val_fn,
        val_interval=args.val_interval,
    ).run_loop()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def create_argparser():
    defaults = dict(
        data_dir="",
        mask_root="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # Validation
        val_data_dir="",
        val_interval=0,        # 0 = disabled
        val_num_samples=16,
        trans_noise_level=0.4,  # SDEdit t0 fraction (used for val and sampling)
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
