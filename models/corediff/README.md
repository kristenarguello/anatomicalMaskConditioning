# 🌀 CoreDiff — Baseline & Mask-Conditioned

Contextual Error-Modulated Generalized Diffusion Model for LDCT denoising ([Gao et al., IEEE TMI 2024](https://arxiv.org/abs/2304.01814)). CoreDiff uses a two-step diffusion process guided by a learned contextual error map. An optional inter-slice context mode stacks adjacent slices as additional input channels, helping the model exploit volumetric continuity.

---

## ⚙️ Variants

Four variants are trained, combining two independent dimensions:

**Inter-slice context (`--context`)** — when enabled, the previous and next QD slices are stacked with the current slice, increasing `in_channels` from 1 to 3.

**Mask conditioning (`--mask_dir`)** — when a mask directory is provided, the **label-map channel** (integer values 1–4, normalized to \[0, 1\]) is appended to the input, adding 1 to `in_channels`.

| Variant | Script | `in_channels` |
|---|---|---|
| Baseline, no context | `run_baseline_no_context.sh` | 1 |
| Baseline, with context | `run_baseline.sh` | 3 |
| Masked, no context | `run_masked_no_context.sh` | 2 |
| Masked, with context | `run_masked.sh` | 4 |

---

## ▶️ Training

```bash
# Baseline with inter-slice context (primary baseline variant)
GPUS=0,1 ./run_baseline.sh [ntfy_topic]

# Baseline without context
GPUS=0,1 ./run_baseline_no_context.sh [ntfy_topic]

# Masked with inter-slice context (primary masked variant)
GPUS=0,1 ./run_masked.sh [ntfy_topic]

# Masked without context
GPUS=0,1 ./run_masked_no_context.sh [ntfy_topic]
```

All variants train for **150 000 iterations**, batch size 8, 25% dose, saving checkpoints every 2500 iterations.

**Environment variables**:

| Variable | Default | Description |
|---|---|---|
| `GPUS` | `0,1` | Comma-separated CUDA device indices |
| `IMG_ROOT` | `./dataset/png_dataset` | PNG image dataset root |
| `MASK_DIR` | `./dataset/multilabel` | Label maps (masked variants only) |

```bash
# Example with custom paths
GPUS=2,3 IMG_ROOT=/data/png MASK_DIR=/data/masks ./run_masked.sh my-topic
```

---

## 📎 Citation

```bibtex
@article{gao2023corediff,
  title={CoreDiff: Contextual Error-Modulated Generalized Diffusion Model for Low-Dose CT Denoising and Generalization},
  author={Gao, Qi and Li, Zilong and Zhang, Junping and Zhang, Yi and Shan, Hongming},
  journal={IEEE Transactions on Medical Imaging},
  volume={43},
  number={2},
  pages={745--759},
  year={2024}
}
```
