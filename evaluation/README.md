# 📊 Evaluation

Unified inference and evaluation pipeline for all model/variant combinations. Produces per-slice metrics, permutation test results, a summary table, and paper figures.

---

## 📁 Files

### Orchestration

| File | Description |
|---|---|
| `run_all_eval.sh` | Runs the full pipeline end-to-end: inference → metrics → permutation tests → summary table. Sends ntfy notifications. |

### Inference

Each script loads the best checkpoint for its model and writes one uint8 prediction PNG per test slice (filename identical to the LDCT input).

| File | Model | Variants |
|---|---|---|
| `infer_redcnn.py` | RED-CNN | `no_mask`, `mask` |
| `infer_corediff.py` | CoreDiff | `corediff_no_mask_noctx`, `corediff_mask_noctx`, `corediff_no_mask_ctx`, `corediff_mask_ctx` |
| `infer_iddpm.py` | iDDPM | `no_mask`, `mask` |
| `infer_segguideddiff.py` | SegGuidedDiff | `no_mask`, `mask` |

### Metrics & Statistics

| File | Description |
|---|---|
| `unified_eval.py` | Computes per-slice **PSNR**, **SSIM**, **RMSE (HU)**, **CNR**, plus boundary- and interior-region PSNR/SSIM using tissue masks. Saves `per_slice_metrics.csv` and fixed qualitative PNG panels. Soft-tissue truncation: \[−160, 240\] HU, data range = 400. |
| `run_permutation_test.py` | **Paired sign-flipping permutation test** (n = 12 patients, 4096 exact permutations). Averages slices within each patient first, then tests whether the mean per-patient delta differs from zero. Outputs a CSV with p-values for each metric. |
| `build_summary_table.py` | Reads all `per_slice_metrics.csv` files, computes mean ± std across slices per variant, and prints a formatted table plus a LaTeX version. |

### Figures

| File | Output |
|---|---|
| `fig_masks.py` | Illustration of the anatomical masks: NDCT + 4 individual binary overlays (top row) and the combined label map fed to the model (bottom row). |
| `fig_qualitative_3models.py` | Qualitative comparison panel: 4 models × 6 columns (LDCT \| NDCT GT \| Baseline \| +Mask \| ROI Baseline \| ROI +Mask). |
| `fig_segguideddiff_tradeoff.py` | SegGuidedDiff trade-off visualization: 6 panels per case (NDCT GT \| LDCT \| Baseline \| +Mask \| signed difference map \| ROI zoom). |

---

## ▶️ Running the Full Pipeline

```bash
./run_all_eval.sh <ntfy_topic> <gpu1> [gpu2]

# Single GPU
./run_all_eval.sh my-topic 0

# Two GPUs (iDDPM and SegGuidedDiff split across both)
./run_all_eval.sh my-topic 0 1
```

This sequentially runs inference for all 10 model/variant combinations, then metrics, then permutation tests, then the summary table. Logs land in `results/logs/`.

---

## 📂 Output Layout

```
results/
├── redcnn_no_mask/
│   ├── predictions/          ← one PNG per test slice
│   └── per_slice_metrics.csv
├── redcnn_mask/
│   └── ...
├── corediff_no_mask_noctx/
│   └── ...
├── corediff_mask_noctx/
│   └── ...
├── corediff_no_mask_ctx/
│   └── ...
├── corediff_mask_ctx/
│   └── ...
├── iddpm_no_mask/
│   └── ...
├── iddpm_mask/
│   └── ...
├── segguideddiff_no_mask/
│   └── ...
├── segguideddiff_mask/
│   └── ...
├── permtest_redcnn.csv
├── permtest_corediff_noctx.csv
├── permtest_corediff_ctx.csv
├── permtest_iddpm.csv
├── permtest_segguideddiff.csv
└── logs/
    ├── infer_redcnn_no_mask.log
    ├── eval_redcnn_no_mask.log
    ├── permtest_redcnn.log
    └── summary_table.log
```

---

## 🔬 Running Steps Individually

```bash
# Inference only (e.g. RED-CNN)
CUDA_VISIBLE_DEVICES=0 python infer_redcnn.py --variant no_mask
CUDA_VISIBLE_DEVICES=0 python infer_redcnn.py --variant mask

# Metrics for one variant
python unified_eval.py \
    --pred_dir  results/redcnn_no_mask/predictions \
    --out_dir   results/redcnn_no_mask \
    --model_name redcnn --variant no_mask

# Permutation test for one model family
python run_permutation_test.py \
    --baseline results/redcnn_no_mask/per_slice_metrics.csv \
    --masked   results/redcnn_mask/per_slice_metrics.csv \
    --model    redcnn \
    --out_csv  results/permtest_redcnn.csv

# Summary table (all variants at once)
python build_summary_table.py --results_root results/
```
