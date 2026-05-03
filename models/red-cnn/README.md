# 🔴 RED-CNN — Baseline & Mask-Conditioned

Residual Encoder-Decoder Convolutional Neural Network for LDCT denoising ([Chen et al., 2017](https://arxiv.org/abs/1702.00288)). The network maps quarter-dose (QD) image patches to normal-dose (NDCT) patches through five encoder convolutions, three residual skip connections, and five decoder transposed convolutions.

---

## ⚙️ Modification for Mask Conditioning

The original model takes a single input channel. For the masked variant, `in_channels` is exposed as a constructor argument (default `1`). The conditioned model receives **2 channels**: the QD patch and the corresponding **label-map patch** (integer values 1–4, normalized to \[0, 1\]).

The residual skip connections are taken from `x[:, :1]` (the QD channel only), so the single-channel output shape is unchanged regardless of `in_channels`. The mask column in the first convolution kernel is **zero-initialised**, so the masked model starts as a functional copy of the baseline.

---

## 📁 Files

| File | Description |
|---|---|
| `networks.py` | `RED_CNN` module — configurable `in_channels` |
| `solver.py` | Training loop, test logic, checkpoint saving |
| `loader.py` | PNG patch DataLoader; supports optional mask channel |
| `main.py` | CLI entry point |
| `measure.py` | PSNR / SSIM helpers |

---

## ▶️ Training

```bash
# Baseline (QD only, in_channels=1)
./run_baseline.sh [ntfy_topic]

# Mask-conditioned (QD + label map, in_channels=2)
./run_masked.sh [ntfy_topic]
```

Both scripts train for **100 epochs**, batch size 128, patch size 64×64, lr 1e-4 with decay.

**Environment variables** (override before running):

| Variable | Default | Description |
|---|---|---|
| `GPU` | `0` | CUDA device index |
| `DATA_PATH` | `./dataset/png_dataset` | PNG image dataset root |
| `MASK_DIR` | `./dataset/multilabel` | Single-channel label maps (masked variant) |

```bash
# Example with custom paths
GPU=1 DATA_PATH=/data/png MASK_DIR=/data/masks ./run_masked.sh my-topic
```

---

## 🧪 Testing

```bash
./run_test.sh
```

Tests both variants sequentially and prints per-slice PSNR/SSIM. Saved images land in `./save/baseline/` and `./save/masked/`.

---

## 📎 Citation

```bibtex
@article{chen2017low,
  title={Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network},
  author={Chen, Hu and Zhang, Yi and Kalra, Mannudeep K and Lin, Feng and Chen, Yang and Liao, Peixi and Zhou, Jiliu and Wang, Ge},
  journal={IEEE Transactions on Medical Imaging},
  volume={36},
  number={12},
  pages={2524--2535},
  year={2017}
}
```
