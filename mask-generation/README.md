# Mask Generation

Generates the anatomical segmentation masks used as conditioning signal for all models.

The pipeline has two steps:

1. **`mask_generation.py`** — runs TotalSegmentator on the QD 1mm DICOM series for each patient, producing one multilabel NIfTI per patient.
2. **`export_masks.py`** — converts those NIfTI files into per-class binary PNG masks, mirroring the PNG dataset layout so that filenames and split assignments match exactly what the model loaders expect.

## Background

Masks are derived from the **quarter-dose (QD)** images rather than the full-dose ground truth, ensuring the conditioning signal is available at inference time when only low-dose scans are accessible.

The segmentation task used is `tissue_4_types` from TotalSegmentator, which produces four tissue classes:

| Class ID | Class name | Tissue |
|---|---|---|
| 1 | `subcutaneous_fat` | Subcutaneous adipose tissue |
| 2 | `torso_fat` | Visceral / torso fat |
| 3 | `skeletal_muscle` | Skeletal muscle |
| 4 | `intermuscular_fat` | Intermuscular fat |

At training time the four per-class binary masks are combined into a single-channel label map (integer values 1–4, normalized to [0, 1] by dividing by 4) and concatenated to the model input as the conditioning channel.

## Requirements

```
totalsegmentator
nibabel
pydicom
natsort
Pillow
numpy
```

## Step 1 — Run TotalSegmentator

### Input layout

```
/path/to/DICOM_ROOT/
├── L001/
│   └── quarter_1mm/
│       ├── IM-0001-0001.dcm
│       └── ...
├── L002/
│   └── quarter_1mm/
│       └── ...
└── ...
```

Multiple roots can be passed (e.g. the 2016 challenge set and the 2020 TCIA extension are stored separately).

### Command

```bash
python mask_generation.py \
    --data_roots /path/to/DICOM_ROOT_2016 /path/to/DICOM_ROOT_2020 \
    --dose quarter \
    --out_root segmentations \
    --device auto
```

`--device auto` picks CUDA if available, then MPS (macOS), then CPU.  
On GPU a single series is processed at a time (`--workers 1`). On CPU up to 4 series run in parallel.

### Outputs

```
segmentations/
├── DICOM_ROOT_2016/
│   ├── L001/quarter_1mm/segmentations.nii.gz   ← multilabel NIfTI
│   ├── L001/quarter_1mm/log.txt
│   ├── L002/quarter_1mm/segmentations.nii.gz
│   └── ...
└── DICOM_ROOT_2020/
    └── ...
```

Each `segmentations.nii.gz` is a 3D multilabel volume in the same voxel space as the input DICOM series. Integer labels follow the class table above; 0 is background.

### Filter

Only series whose DICOM `ConvolutionKernel` tag contains `B30` are processed. Other kernels are skipped automatically.

---

## Step 2 — Export to PNG masks

Converts the NIfTI volumes to per-class binary PNG masks with filenames and directory structure matching the PNG dataset used for training.

### Input layout (PNG dataset)

```
/path/to/png_dataset/
├── train/1mm/QD/*.png
├── train/1mm/FD/*.png
├── val/1mm/QD/*.png
└── test/1mm/QD/*.png
```

### Command

```bash
python export_masks.py \
    --seg_root  segmentations \
    --png_root  /path/to/png_dataset \
    --mask_dir  masks
```

### Outputs

```
masks/
├── subcutaneous_fat/
│   ├── train/1mm/QD/L001_QD_1_1_CT_0001.png
│   ├── train/1mm/QD/L001_QD_1_1_CT_0002.png
│   └── ...
├── torso_fat/
│   └── ...
├── skeletal_muscle/
│   └── ...
└── intermuscular_fat/
    └── ...
```

Each PNG is an 8-bit grayscale image: `255` = class present, `0` = background. Filenames match the QD PNG filenames exactly, and each patient lands in the split (`train`/`val`/`test`) it belongs to in the PNG dataset.

The `masks/` directory is what you pass as `--mask_dir` (RED-CNN) or `mask_dir` (CoreDiff) when training the conditioned model variants.

### Slice alignment

The NIfTI is reoriented to RAS canonical orientation (z = inferior → superior) before slicing. The QD PNG files are naturally sorted (natsort), which also gives inferior → superior order. The z-th NIfTI slice is therefore exported with the z-th QD PNG filename. The script validates that slice counts match and skips patients with a mismatch, printing a diagnostic message.

---

## Full pipeline

```bash
# 1. Run TotalSegmentator on QD 1mm DICOM series
python mask_generation.py \
    --data_roots /data/LDCT_2016 /data/LDCT_2020 \
    --out_root   segmentations

# 2. Export NIfTI → model-ready PNG masks
python export_masks.py \
    --seg_root  segmentations \
    --png_root  /data/png_dataset \
    --mask_dir  masks
```
