"""
Figure: Qualitative comparison — 4 models × 6 columns
Columns: LDCT | NDCT GT | Baseline | +Mask | ROI Baseline | ROI +Mask
Rows: CoreDiff, iDDPM, SegGuidedDiff, RED-CNN (one case each)

Usage:
    python fig_qualitative_3models.py

Edit CASE_PER_MODEL to pick specific cases per model.
Edit ROI_PER_MODEL to pin the crop box, or leave None for auto-detect.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_RESULTS  = os.environ.get("RESULTS_ROOT", os.path.join(_HERE, "results"))
QUAL_DIR  = os.path.join(_RESULTS, "qualitative")
PRED_DIR  = _RESULTS
OUT_PATH  = os.path.join(_RESULTS, "fig_qualitative_3models.pdf")
OUT_PNG   = os.path.join(_RESULTS, "fig_qualitative_3models.png")

# One case folder per model row.  Set to None to auto-pick (first available).
# corediff_ctx: L014 is missing — default falls back to L058.
CASE_PER_MODEL = {
    "corediff_noctx": "L058_QD_1_1_CT_0037",
    "corediff_ctx":   "L058_QD_1_1_CT_0037",
    "iddpm":          "L058_QD_1_1_CT_0037",
    "segguideddiff":  "L058_QD_1_1_CT_0037",
    "redcnn":         "L058_QD_1_1_CT_0037",
}

# ROI crop: (row_start, row_end, col_start, col_end) in pixel coords (0-512).
# Set to None to auto-detect the patch with the largest mask-vs-baseline diff.
ROI_PER_MODEL = {
    "corediff_noctx": None,
    "corediff_ctx":   None,
    "iddpm":          None,
    "segguideddiff":  None,
    "redcnn":         None,
}
ROI_SIZE   = 128   # square crop size used when auto-detecting
ROI_STRIDE = 16    # sliding-window stride for auto-detect

# Display
VMIN, VMAX = 0, 255
CMAP_GRAY  = "gray"
DPI        = 300
ROI_RECT_COLOR = "red"

ROW_LABELS = {
    "corediff_noctx": "CoreDiff\n(no ctx)",
    "corediff_ctx":   "CoreDiff\n(ctx)",
    "iddpm":          "iDDPM",
    "segguideddiff":  "SegGuidedDiff",
    "redcnn":         "RED-CNN",
}

FILE_LDCT = "ldct.png"
FILE_GT   = "ndct_gt.png"

# Values starting with "/" are prediction directories: load as {dir}/{case}.png
# Plain strings are filenames inside QUAL_DIR/{case}/
BASELINE_FILE = {
    "corediff_noctx": "corediff_no_mask_noctx_pred.png",
    "corediff_ctx":   f"{PRED_DIR}/corediff_no_mask_ctx/predictions",
    "iddpm":          "iddpm_no_mask_pred.png",
    "segguideddiff":  "segguideddiff_no_mask_pred.png",
    "redcnn":         "redcnn_no_mask_pred.png",
}
MASK_FILE = {
    "corediff_noctx": "corediff_mask_noctx_pred.png",
    "corediff_ctx":   f"{PRED_DIR}/corediff_mask_ctx/predictions",
    "iddpm":          "iddpm_mask_pred.png",
    "segguideddiff":  "segguideddiff_mask_pred.png",
    "redcnn":         "redcnn_mask_pred.png",
}

# Cases where ctx predictions were not generated (skip when auto-picking)
CTX_MISSING = {"L014_QD_1_1_CT_0001"}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def all_cases():
    return sorted(os.listdir(QUAL_DIR))


def pick_case(model, pref):
    if pref is not None:
        return pref
    cases = all_cases()
    if model == "corediff_ctx":
        cases = [c for c in cases if c not in CTX_MISSING]
    return cases[0]


def resolve_img(spec, case, case_dir):
    """Return the image path for a given file spec and case."""
    if spec.startswith("/"):
        return os.path.join(spec, f"{case}.png")
    return os.path.join(case_dir, spec)


def auto_roi(baseline, masked, size=ROI_SIZE, stride=ROI_STRIDE):
    """Return (r0, r1, c0, c1) for the patch maximising mean |masked - baseline|."""
    diff = np.abs(masked - baseline)
    h, w = diff.shape
    best_score, best_box = -1, (0, size, 0, size)
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            score = diff[r:r+size, c:c+size].mean()
            if score > best_score:
                best_score = score
                best_box = (r, r + size, c, c + size)
    return best_box


def draw_roi_rect(ax, box, color=ROI_RECT_COLOR, lw=1.5):
    r0, r1, c0, c1 = box
    ax.add_patch(patches.Rectangle(
        (c0, r0), c1 - c0, r1 - r0,
        linewidth=lw, edgecolor=color, facecolor="none"
    ))

# ── LAYOUT ────────────────────────────────────────────────────────────────────
# Columns 0-3: full-resolution images
# Columns 4-5: matched ROI crops (baseline vs +mask) for direct comparison

models     = ["corediff_ctx", "redcnn"]
# models = ["corediff_noctx",  "iddpm", "segguideddiff"]
col_titles = ["LDCT", "NDCT (GT)", "Baseline", "+Mask", "ROI  Baseline", "ROI  +Mask"]
n_rows = len(models)
n_cols = len(col_titles)

# Cols 4-5 are crops so can be narrower; use width_ratios
width_ratios = [2, 2, 2, 2, 1.1, 1.1]

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(sum(width_ratios) * 1.15, n_rows * 2.1),
    gridspec_kw={"wspace": 0.04, "hspace": 0.06, "width_ratios": width_ratios},
)

for row_idx, model in enumerate(models):
    case     = pick_case(model, CASE_PER_MODEL[model])
    case_dir = os.path.join(QUAL_DIR, case)

    ldct     = load_gray(os.path.join(case_dir, FILE_LDCT))
    gt       = load_gray(os.path.join(case_dir, FILE_GT))
    baseline = load_gray(resolve_img(BASELINE_FILE[model], case, case_dir))
    masked   = load_gray(resolve_img(MASK_FILE[model],     case, case_dir))

    roi = ROI_PER_MODEL[model]
    if roi is None:
        roi = auto_roi(baseline, masked)
    r0, r1, c0, c1 = roi

    # ── Full-image columns (0-3) ──────────────────────────────────────────────
    for col_idx, img in enumerate([ldct, gt, baseline, masked]):
        ax = axes[row_idx, col_idx]
        ax.imshow(img, cmap=CMAP_GRAY, vmin=VMIN, vmax=VMAX, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        # Red rectangle on both Baseline (2) and +Mask (3)
        if col_idx in (2, 3):
            draw_roi_rect(ax, roi)

    # ── ROI crop columns (4-5) ────────────────────────────────────────────────
    crop_kw = dict(cmap=CMAP_GRAY, vmin=VMIN, vmax=VMAX, interpolation="nearest")

    axes[row_idx, 4].imshow(baseline[r0:r1, c0:c1], **crop_kw)
    axes[row_idx, 4].set_xticks([])
    axes[row_idx, 4].set_yticks([])

    axes[row_idx, 5].imshow(masked[r0:r1, c0:c1], **crop_kw)
    axes[row_idx, 5].set_xticks([])
    axes[row_idx, 5].set_yticks([])

    # Thin border on crop panels to visually separate them from full images
    for col_idx in (4, 5):
        for spine in axes[row_idx, col_idx].spines.values():
            spine.set_edgecolor(ROI_RECT_COLOR)
            spine.set_linewidth(1.2)

    # Row label
    axes[row_idx, 0].set_ylabel(
        ROW_LABELS[model], fontsize=11, fontweight="bold", labelpad=4
    )

    # Case ID annotation
    axes[row_idx, 0].annotate(
        case, xy=(2, 510), fontsize=8, color="white",
        xycoords="data", va="bottom",
    )

# Column titles
for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, fontsize=11, pad=4)

fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
fig.savefig(OUT_PNG,  dpi=DPI, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
print(f"Saved: {OUT_PNG}")
print("\nCases used:")
for model in models:
    print(f"  {model}: {pick_case(model, CASE_PER_MODEL[model])}")
print("\nROIs used:")
for model in models:
    case     = pick_case(model, CASE_PER_MODEL[model])
    case_dir = os.path.join(QUAL_DIR, case)
    bl = load_gray(resolve_img(BASELINE_FILE[model], case, case_dir))
    mk = load_gray(resolve_img(MASK_FILE[model],     case, case_dir))
    roi = ROI_PER_MODEL[model] or auto_roi(bl, mk)
    print(f"  {model}: rows {roi[0]}-{roi[1]}, cols {roi[2]}-{roi[3]}")
