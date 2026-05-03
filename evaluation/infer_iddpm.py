#!/usr/bin/env python3
"""
iDDPM (improved-diffusion) inference adapter.

Loads the EMA checkpoint from improved-diffusion-ct/runs/{no_mask,with_mask}/
and runs SDEdit-style inference on all test slices via image_sample.py,
then converts the uint16 outputs to uint8 for unified_eval.py.

Usage:
  python infer_iddpm.py --variant no_mask
  python infer_iddpm.py --variant mask
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE            = os.path.dirname(os.path.abspath(__file__))
IDDPM_CODE_ROOT  = os.path.join(_HERE, "..", "models", "iddpm")
IDDPM_CKPT_ROOT  = os.path.join(_HERE, "..", "models", "iddpm")
DATASET_ROOT     = os.environ.get("DATASET_ROOT", os.path.join(_HERE, "..", "dataset"))
LDCT_DIR         = os.path.join(DATASET_ROOT, "png_dataset", "test", "1mm", "QD")
MASK_ROOT        = os.path.join(DATASET_ROOT, "multilabel")
RESULTS_ROOT     = os.environ.get("RESULTS_ROOT", os.path.join(_HERE, "results"))

VARIANTS = {
    "no_mask": {
        "run_dir": os.path.join(IDDPM_CKPT_ROOT, "runs", "no_mask"),
        "use_mask": False,
    },
    "mask": {
        "run_dir": os.path.join(IDDPM_CKPT_ROOT, "runs", "with_mask"),
        "use_mask": True,
    },
}

# Model architecture (must match training)
MODEL_FLAGS = [
    "--image_size",             "512",
    "--image_channels",         "1",
    "--num_channels",           "64",
    "--num_res_blocks",         "2",
    "--attention_resolutions",  "32,16,8",
    "--learn_sigma",            "True",
    "--diffusion_steps",        "1000",
    "--noise_schedule",         "cosine",
    "--rescale_learned_sigmas", "True",
]


def uint16_to_uint8(src_dir: str, dst_dir: str):
    """Convert iDDPM uint16 _pred.png outputs to uint8 named as LDCT files."""
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if not fname.endswith("_pred.png"):
            continue
        src = os.path.join(src_dir, fname)
        # stem is the original QD name; image_sample.py saves {stem}_pred.png
        stem = fname[: -len("_pred.png")]
        out_fname = stem + ".png"
        dst = os.path.join(dst_dir, out_fname)
        if os.path.exists(dst):
            continue
        img = Image.open(src)
        arr = np.array(img, dtype=np.float32)
        if arr.max() > 256:          # uint16
            arr = arr / 257.0        # → [0, 255]
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(dst)


def run(variant: str, gpus: str = "0", batch_size: int = 32):
    cfg = VARIANTS[variant]
    run_dir = cfg["run_dir"]
    use_mask = cfg["use_mask"]

    # Find latest EMA checkpoint
    ema_files = sorted(
        f for f in os.listdir(run_dir) if f.startswith("ema_0.9999_")
    )
    if not ema_files:
        print(f"[ERROR] no EMA checkpoint in {run_dir}", file=sys.stderr)
        sys.exit(1)
    ckpt_name = ema_files[-1]
    ckpt_path = os.path.join(run_dir, ckpt_name)
    gpu_list = [g.strip().replace("cuda:", "") for g in gpus.split(",") if g.strip()]
    world_size = len(gpu_list)
    print(f"[infer_iddpm] variant={variant}  checkpoint={ckpt_name}  gpus={gpu_list}")

    raw_out_dir  = os.path.join(RESULTS_ROOT, f"iddpm_{variant}", "raw_uint16")
    pred_out_dir = os.path.join(RESULTS_ROOT, f"iddpm_{variant}", "predictions")
    os.makedirs(raw_out_dir,  exist_ok=True)
    os.makedirs(pred_out_dir, exist_ok=True)

    base_env = os.environ.copy()
    base_env["OPENAI_LOGDIR"] = raw_out_dir
    base_env["PYTHONPATH"]    = IDDPM_CODE_ROOT + ":" + base_env.get("PYTHONPATH", "")

    base_cmd = [
        sys.executable,
        os.path.join(IDDPM_CODE_ROOT, "scripts", "image_sample.py"),
        "--model_path",  ckpt_path,
        "--ldct_dir",    LDCT_DIR,
        "--out_dir",     raw_out_dir,
        "--use_mask",    "True" if use_mask else "False",
        "--trans_noise_level", "0.4",
        "--clip_denoised",     "True",
        "--batch_size",        str(batch_size),
        "--use_ddim",          "False",
        "--world_size",        str(world_size),
    ]
    if use_mask:
        base_cmd += ["--mask_root", MASK_ROOT]
    base_cmd += MODEL_FLAGS

    print(f"[infer_iddpm] running image_sample.py on {world_size} GPU(s) …")
    if world_size == 1:
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        ret = subprocess.run(base_cmd + ["--rank", "0"], env=env)
        if ret.returncode != 0:
            print(f"[ERROR] image_sample.py exited with {ret.returncode}", file=sys.stderr)
            sys.exit(ret.returncode)
    else:
        procs = []
        for rank, gpu in enumerate(gpu_list):
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            procs.append(subprocess.Popen(base_cmd + ["--rank", str(rank)], env=env))
        for proc in procs:
            ret = proc.wait()
            if ret != 0:
                print(f"[ERROR] image_sample.py subprocess exited with {ret}", file=sys.stderr)
                sys.exit(ret)

    print(f"[infer_iddpm] converting uint16 → uint8 …")
    uint16_to_uint8(raw_out_dir, pred_out_dir)
    print(f"[infer_iddpm] done → {pred_out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["no_mask", "mask"], required=True)
    p.add_argument("--gpus", default="0",
                   help="Comma-separated GPU IDs, e.g. '3' or '3,0' for two GPUs")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    run(args.variant, args.gpus, args.batch_size)
