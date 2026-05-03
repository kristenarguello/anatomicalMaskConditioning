# 🌫️ iDDPM — Baseline & Mask-Conditioned

Improved Denoising Diffusion Probabilistic Model ([Nichol & Dhariwal, ICML 2021](https://arxiv.org/abs/2102.09672)) adapted for LDCT denoising via the **SDEdit** paradigm. At inference, the quarter-dose (QD) image is partially noised to timestep **t₀ = 0.4 T** and then denoised by the model, translating it toward the normal-dose (NDCT) distribution without paired supervision on the diffusion trajectory.

---

## ⚙️ Modification for Mask Conditioning

The baseline UNet takes a **single image channel**. For mask conditioning, the **label map** (integer values 1–4, normalized to \[0, 1\]) is appended as a **third input channel** alongside the QD image, controlled by `--use_mask True --mask_root <path>`. The UNet's input convolution width is increased accordingly; the mask weights are zero-initialised.

The SDEdit noise level is set with `--trans_noise_level 0.4` (applies during inference / validation; training is unconditional).

---

## 🚀 Installation

```bash
pip install -e .
```

This installs the `improved_diffusion` Python package that the training scripts depend on.

---

## ▶️ Training

```bash
# Baseline (QD only)
GPUS=0,1 ./run_no_mask.sh [ntfy_topic]

# Mask-conditioned (QD + label map)
GPUS=0,1 ./run_with_mask.sh [ntfy_topic]
```

Both scripts use `mpiexec` for multi-GPU training. They **auto-detect and resume** from the latest checkpoint in `LOG_DIR`.

Key hyperparameters: image size 512, cosine noise schedule, 1000 diffusion steps, lr 1.5e-4, 100k annealing steps.

**Environment variables**:

| Variable | Default | Description |
|---|---|---|
| `GPUS` | `0,1` | Comma-separated CUDA device indices |
| `DATA_DIR` | `./dataset/png_dataset/train,./dataset/png_dataset/val` | Training image dirs |
| `VAL_DIR` | `./dataset/png_dataset/test` | Validation image dir |
| `MASK_ROOT` | `./dataset/multilabel` | Label maps (masked variant only) |
| `LOG_DIR` | `./runs/no_mask` or `./runs/with_mask` | Checkpoint and log directory |

```bash
# Example with custom paths
GPUS=2,3 DATA_DIR=/data/png/train,/data/png/val \
         MASK_ROOT=/data/masks \
         LOG_DIR=/runs/iddpm_masked \
         ./run_with_mask.sh my-topic
```

---

## 📎 Citation

```bibtex
@inproceedings{nichol2021improved,
  title={Improved Denoising Diffusion Probabilistic Models},
  author={Nichol, Alexander Quinn and Dhariwal, Prafulla},
  booktitle={International Conference on Machine Learning},
  pages={8162--8171},
  year={2021}
}2
```
