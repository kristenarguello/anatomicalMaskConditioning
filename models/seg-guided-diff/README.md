# 🎯 SegGuidedDiff — Baseline & Mask-Conditioned

Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models ([Konz et al., MICCAI 2024](https://arxiv.org/abs/2402.05210)). SegGuidedDiff conditions a DDPM at **every denoising step** by concatenating a segmentation mask to the noisy image, giving the model direct spatial guidance throughout the reverse process.

---

## ⚙️ Adaptation for LDCT Denoising

The model is adapted for image-to-image translation via the **SDEdit** paradigm at t₀ = **0.2 T** — a shallower noise injection than iDDPM, taking advantage of the explicit per-step mask conditioning to guide the reverse trajectory more tightly.

**Baseline** — standard DDPM trained without segmentation guidance; the QD image is noised to t₀ = 0.2 T and denoised unconditionally.

**Masked variant** — the label map (**5 classes**: background + subcutaneous fat, torso fat, skeletal muscle, intermuscular fat) is appended to the noisy image at each denoising step via `--segmentation_channel_mode single --num_segmentation_classes 5`. The single-channel mask encodes class identity as integer values 0–4.

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

```bash
# Baseline
GPUS=0,1 ./run_baseline.sh

# Mask-conditioned
GPUS=0,1 IMG_DIR=/data/png_dataset SEG_DIR=/data/masks ./run_masked.sh
```

Both variants train for **200 epochs**, batch size 8, lr 5e-5, image size 512, DDPM sampling.

**Environment variables**:

| Variable | Default | Description |
|---|---|---|
| `GPUS` | `0,1` | Comma-separated CUDA device indices |
| `IMG_DIR` | `dataset/png_dataset` | PNG image dataset root |
| `SEG_DIR` | `dataset/multilabel` | Single-channel label maps (masked variant only) |

---

## 📎 Citation

```bibtex
@inproceedings{konz2024segguideddiffusion,
  title={Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models},
  author={Nicholas Konz and Yuwen Chen and Haoyu Dong and Maciej A. Mazurowski},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2024}
}
```
