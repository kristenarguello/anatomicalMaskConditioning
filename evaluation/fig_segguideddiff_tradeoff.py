"""
Figure: SegGuidedDiff trade-off — structure improvement vs texture degradation
Layout: 1 row × 6 panels, or 2 rows with multiple cases

Panel order:
  NDCT GT | LDCT | Baseline | +Mask | Diff(+Mask − Baseline) | ROI zoom

The diff panel uses a signed diverging colormap (RdBu_r):
  - Blue  = +Mask darker than baseline  (less noise / structure improved)
  - Red   = +Mask brighter (more artifacts / texture degraded)

Usage:
    python fig_segguideddiff_tradeoff.py

To show multiple cases (one per row), add more entries to CASES.
To pin an ROI, edit ROI_BOXES.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────

QUAL_DIR = "/mnt/G-SSD/kristen/results/qualitative"
OUT_PATH = "/mnt/G-SSD/kristen/results/fig_segguideddiff_tradeoff.pdf"
OUT_PNG  = "/mnt/G-SSD/kristen/results/fig_segguideddiff_tradeoff.png"

# Cases to show — each entry becomes one row.
# Set to None to use first available case.
CASES = [None, None]   # e.g. ["L014_QD_1_1_CT_0001", "L058_QD_1_1_CT_0037"]

# ROI crop per row: (r0, r1, c0, c1).  None = auto-detect.
ROI_BOXES = [None, None]
ROI_SIZE   = 128
ROI_STRIDE = 16

# Diverging diff scale: symmetric clamp at ±DIFF_CLAMP gray levels.
# Increase to show subtler differences; decrease to saturate more.
DIFF_CLAMP = 30

# Display
VMIN, VMAX = 0, 255
CMAP_GRAY  = "gray"
CMAP_DIFF  = "RdBu_r"
DPI        = 300

BASELINE_FILE = "segguideddiff_no_mask_pred.png"
MASK_FILE     = "segguideddiff_mask_pred.png"
FILE_LDCT     = "ldct.png"
FILE_GT       = "ndct_gt.png"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def all_cases():
    return sorted(os.listdir(QUAL_DIR))


def resolve_cases(cases):
    available = all_cases()
    return [c if c is not None else available[i % len(available)]
            for i, c in enumerate(cases)]


def auto_roi(baseline, masked, size=ROI_SIZE, stride=ROI_STRIDE):
    diff = np.abs(masked - baseline)
    h, w = diff.shape
    best, best_box = -1, (0, size, 0, size)
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            score = diff[r:r+size, c:c+size].mean()
            if score > best:
                best = score
                best_box = (r, r + size, c, c + size)
    return best_box


def draw_rect(ax, box, color="red", lw=1.5):
    r0, r1, c0, c1 = box
    ax.add_patch(patches.Rectangle(
        (c0, r0), c1 - c0, r1 - r0,
        linewidth=lw, edgecolor=color, facecolor="none"
    ))


# ── MAIN ──────────────────────────────────────────────────────────────────────

cases = resolve_cases(CASES)
n_rows = len(cases)
col_titles = ["NDCT (GT)", "LDCT", "Baseline", "+Mask", "Diff (+Mask − Baseline)", "ROI Zoom"]
n_cols = len(col_titles)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(n_cols * 2.0, n_rows * 2.1),
    gridspec_kw={"wspace": 0.05, "hspace": 0.45},
)
if n_rows == 1:
    axes = axes[np.newaxis, :]   # always 2-D indexing

for row_idx, case in enumerate(cases):
    case_dir = os.path.join(QUAL_DIR, case)

    gt       = load_gray(os.path.join(case_dir, FILE_GT))
    ldct     = load_gray(os.path.join(case_dir, FILE_LDCT))
    baseline = load_gray(os.path.join(case_dir, BASELINE_FILE))
    masked   = load_gray(os.path.join(case_dir, MASK_FILE))
    diff     = masked - baseline   # signed

    roi = ROI_BOXES[row_idx] if row_idx < len(ROI_BOXES) else None
    if roi is None:
        roi = auto_roi(baseline, masked)
    r0, r1, c0, c1 = roi

    # ── cols 0-3: full images ────────────────────────────────────────────────
    for col_idx, img in enumerate([gt, ldct, baseline, masked]):
        ax = axes[row_idx, col_idx]
        ax.imshow(img, cmap=CMAP_GRAY, vmin=VMIN, vmax=VMAX, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 3:
            draw_rect(ax, roi)

    # ── col 4: full-image signed diff ────────────────────────────────────────
    ax_diff = axes[row_idx, 4]
    im = ax_diff.imshow(
        diff, cmap=CMAP_DIFF,
        vmin=-DIFF_CLAMP, vmax=DIFF_CLAMP,
        interpolation="lanczos"
    )
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])
    draw_rect(ax_diff, roi, color="black")

    # Colorbar below diff panel (only on last row to save space)
    if row_idx == n_rows - 1:
        divider = make_axes_locatable(ax_diff)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Δ gray level", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    # ── col 5: ROI zoom of signed diff ───────────────────────────────────────
    ax_zoom = axes[row_idx, 5]
    diff_crop = diff[r0:r1, c0:c1]
    ax_zoom.imshow(
        diff_crop, cmap=CMAP_DIFF,
        vmin=-DIFF_CLAMP, vmax=DIFF_CLAMP,
        interpolation="nearest"
    )
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    # Row label
    axes[row_idx, 0].set_ylabel(case, fontsize=8, labelpad=3)

# Column titles
for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, fontsize=11, pad=4)

# Annotation: structural improvement = blue; texture degradation = red
fig.text(
    0.98, 0.5,
    "Blue = −Δ (improved structure)\nRed = +Δ (texture degraded)",
    ha="right", va="center", fontsize=9, style="italic",
    transform=fig.transFigure,
)

fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
fig.savefig(OUT_PNG,  dpi=DPI, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
print(f"Saved: {OUT_PNG}")
print("Cases used:", cases)
print("ROIs used (after auto-detect if needed):")
for row_idx, case in enumerate(cases):
    case_dir = os.path.join(QUAL_DIR, case)
    baseline = load_gray(os.path.join(case_dir, BASELINE_FILE))
    masked   = load_gray(os.path.join(case_dir, MASK_FILE))
    roi = ROI_BOXES[row_idx] if row_idx < len(ROI_BOXES) else None
    if roi is None:
        roi = auto_roi(baseline, masked)
    print(f"  {case}: rows {roi[0]}-{roi[1]}, cols {roi[2]}-{roi[3]}")
