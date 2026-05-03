# 🧠 Anatomical Mask Conditioning for LDCT Denoising

Does providing anatomical segmentation masks as spatial conditioning improve low-dose CT (LDCT) denoising? This repository contains the code for a controlled study across four architectures — from a classical CNN to diffusion models — each trained as a **baseline** (image-only) and a **mask-conditioned** variant on the AAPM-Mayo 2016/2020 dataset.

---

## 📌 Overview

Anatomical masks are derived from the **quarter-dose (QD) 1mm** images using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (`tissue_4_types` task), producing four tissue classes: subcutaneous fat, torso fat, skeletal muscle, and intermuscular fat. The four per-class binary maps are merged into a **single-channel label map** (values 1–4, normalized to \[0, 1\]) and concatenated to each model's input as an additional conditioning channel.

Evaluation uses **paired sign-flipping permutation tests** at the patient level (n = 12, 4096 permutations) on per-slice PSNR, SSIM, RMSE (HU), CNR, and region-specific metrics (boundary vs. interior).

---

## 🗂️ Repository Structure

```
anatomicalMaskConditioning/
├── mask-generation/          # TotalSegmentator pipeline → model-ready PNG masks
│   ├── mask_generation.py    # Step 1: run TotalSegmentator on QD 1mm DICOM
│   ├── export_masks.py       # Step 2: NIfTI → per-class PNG masks
│   ├── conversion.py         # Utility: multilabel NIfTI ↔ per-class PNGs
│   └── README.md
├── models/
│   ├── red-cnn/              # Residual encoder-decoder CNN (Chen et al., 2017)
│   ├── corediff/             # Contextual error-modulated diffusion (Gao et al., 2024)
│   ├── iddpm/                # Improved DDPM + SDEdit (Nichol & Dhariwal, 2021)
│   └── seg-guided-diff/      # Segmentation-guided diffusion (Konz et al., 2024)
└── evaluation/               # Unified inference, metrics, permutation tests, figures
    ├── run_all_eval.sh        # Orchestrates the full evaluation pipeline
    ├── infer_*.py             # Per-model inference scripts
    ├── unified_eval.py        # Per-slice PSNR / SSIM / RMSE / CNR
    ├── run_permutation_test.py
    ├── build_summary_table.py
    └── fig_*.py               # Figure generation
```

---

## 🧬 Mask Generation

```bash
# Step 1 — run TotalSegmentator on QD 1mm DICOM series
python mask-generation/mask_generation.py \
    --data_roots /data/LDCT_2016 /data/LDCT_2020 \
    --out_root   segmentations

# Step 2 — convert NIfTI segmentations → model-ready PNG masks
python mask-generation/export_masks.py \
    --seg_root  segmentations \
    --png_root  /data/png_dataset \
    --mask_dir  masks
```

See [mask-generation/README.md](mask-generation/README.md) for full instructions, input/output layouts, and the slice-alignment protocol.

---

## 🤖 Models

| Model | Architecture | Conditioning mechanism |
|---|---|---|
| **RED-CNN** | Residual encoder-decoder CNN | QD + label map as 2-channel input |
| **CoreDiff** | Contextual error-modulated diffusion | Extra input channel (±inter-slice context) |
| **iDDPM** | Improved DDPM + SDEdit (t₀ = 0.4 T) | Label map as 3rd input channel |
| **SegGuidedDiff** | Segmentation-guided DDPM + SDEdit (t₀ = 0.2 T) | Mask appended at each denoising step |

Each model directory contains its own README with training commands and modification details.

---

## 📊 Evaluation

The full evaluation pipeline is orchestrated by a single script:

```bash
cd evaluation/
./run_all_eval.sh <ntfy_topic> <gpu>
# e.g.
./run_all_eval.sh my-topic 0
```

This will:
1. Run inference for all model/variant combinations
2. Compute per-slice PSNR, SSIM, RMSE (HU), CNR, boundary/interior regional metrics
3. Run paired permutation tests (baseline vs. masked, per model family)
4. Build a unified summary table

Results land in `evaluation/results/`.

---

## ▶️ Quick Start

```bash
# 1. Generate masks
cd mask-generation/
python mask_generation.py --data_roots /data/LDCT_2016 --out_root segmentations
python export_masks.py --seg_root segmentations --png_root /data/png_dataset --mask_dir masks

# 2. Train a model (example: RED-CNN masked variant)
cd ../models/red-cnn/
DATA_PATH=/data/png_dataset MASK_DIR=/data/masks ./run_masked.sh

# 3. Evaluate all models
cd ../../evaluation/
./run_all_eval.sh my-topic 0
```

---

## 📦 Dataset

**AAPM-Mayo LDCT 2016 Grand Challenge** + **2020 TCIA extension**  
Quarter-dose (QD) 1mm reconstruction, B30 kernel. 12 test patients (~1906 test slices).

- 2016 data: https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/  
- 2020 TCIA extension: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026
