"""
Figure: segmentation masks used as conditioning signal.
Layout:
  Row 1 (top): NDCT GT + 4 individual binary mask overlays (for readability)
  Row 2 (bottom): actual combined label map the model receives as input
                  (integer values 0-4, normalized to [0,1] by /N_CLASSES)

Class encoding (matches paper description):
  0 = background
  1 = subcutaneous fat
  2 = torso fat
  3 = skeletal muscle
  4 = intermuscular fat

Usage:
    python fig_masks.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_RESULTS  = os.environ.get("RESULTS_ROOT", os.path.join(_HERE, "results"))
MASK_BASE = os.path.join(
    os.environ.get("DATASET_ROOT", os.path.join(_HERE, "..", "dataset")), "multilabel"
)
QUAL_DIR  = os.path.join(_RESULTS, "qualitative")
OUT_PATH  = os.path.join(_RESULTS, "fig_masks.pdf")
OUT_PNG   = os.path.join(_RESULTS, "fig_masks.png")

CASE       = "L058_QD_1_1_CT_0037"
MASK_SPLIT = "test/1mm/QD"

N_CLASSES  = 4   # label map normalized by dividing by this value

# (tissue_dir, overlay_color, class_value)
TISSUES = {
    "Skeletal\nMuscle":   ("skeletal_muscle",   "#E63946", 3),
    "Subcutaneous\nFat":  ("subcutaneous_fat",  "#457B9D", 1),
    "Torso\nFat":         ("torso_fat",         "#2A9D8F", 2),
    "Intermuscular\nFat": ("intermuscular_fat", "#E9C46A", 4),
}
OVERLAY_ALPHA = 0.55

VMIN, VMAX = 0, 255
DPI = 300

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def mask_path(tissue_dir):
    return os.path.join(MASK_BASE, tissue_dir, MASK_SPLIT, f"{CASE}.png")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

ct = load_gray(os.path.join(QUAL_DIR, CASE, "ndct_gt.png"))

# Build combined integer label map (0 = background, 1-4 = tissues)
label_map = np.zeros(ct.shape, dtype=np.int32)
binary_masks = {}
for label, (tissue_dir, color, class_val) in TISSUES.items():
    bm = load_gray(mask_path(tissue_dir)) / 255.0   # binary {0, 1}
    binary_masks[label] = (bm, color)
    label_map[bm > 0.5] = class_val

# Normalized label map as the model receives it
label_map_norm = label_map / N_CLASSES   # values in {0, 0.25, 0.5, 0.75, 1.0}

# ── COLORMAP for label map ────────────────────────────────────────────────────
# One color per class (index 0-4)
class_colors = ["#000000",   # 0 background  — black
                "#457B9D",   # 1 subcutaneous fat
                "#2A9D8F",   # 2 torso fat
                "#E63946",   # 3 skeletal muscle
                "#E9C46A"]   # 4 intermuscular fat
cmap_label = ListedColormap(class_colors)
bounds     = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm_label = BoundaryNorm(bounds, cmap_label.N)

# ── FIGURE ────────────────────────────────────────────────────────────────────

n_tissues = len(TISSUES)
fig = plt.figure(figsize=(n_tissues * 2.0 + 2.0, 5.5))

gs = fig.add_gridspec(
    2, n_tissues + 1,
    height_ratios=[1, 1.1],
    hspace=0.10, wspace=0.04,
)

# ── Row 0: NDCT GT + individual mask overlays ─────────────────────────────────

ax_ct = fig.add_subplot(gs[0, 0])
ax_ct.imshow(ct, cmap="gray", vmin=VMIN, vmax=VMAX, interpolation="lanczos")
ax_ct.set_title("NDCT (GT)", fontsize=11, pad=4)
ax_ct.set_xticks([]); ax_ct.set_yticks([])

for col_idx, (label, (bm, color)) in enumerate(binary_masks.items(), start=1):
    ax = fig.add_subplot(gs[0, col_idx])
    ax.imshow(ct, cmap="gray", vmin=VMIN, vmax=VMAX, interpolation="lanczos")
    rgba = np.zeros((*bm.shape, 4))
    r, g, b = hex_to_rgb(color)
    rgba[..., 0] = r; rgba[..., 1] = g; rgba[..., 2] = b
    rgba[..., 3] = bm * OVERLAY_ALPHA
    ax.imshow(rgba, interpolation="nearest")
    ax.set_title(label, fontsize=11, pad=4)
    ax.set_xticks([]); ax.set_yticks([])

# ── Row 1: actual combined label map ──────────────────────────────────────────

ax_lm = fig.add_subplot(gs[1, :])
ax_lm.imshow(ct, cmap="gray", vmin=VMIN, vmax=VMAX, interpolation="lanczos")

# Build RGBA overlay: background (class 0) stays transparent
combined_rgba = np.zeros((*label_map.shape, 4), dtype=np.float32)
for class_val, color in enumerate(class_colors):
    if class_val == 0:
        continue
    mask = label_map == class_val
    r, g, b = hex_to_rgb(color)
    combined_rgba[mask, 0] = r
    combined_rgba[mask, 1] = g
    combined_rgba[mask, 2] = b
    combined_rgba[mask, 3] = OVERLAY_ALPHA

ax_lm.imshow(combined_rgba, interpolation="nearest")
ax_lm.set_title(
    f"Combined label map (model input, normalized to [0,1] by ÷{N_CLASSES})",
    fontsize=11, pad=4,
)
ax_lm.set_xticks([]); ax_lm.set_yticks([])

class_labels = {1: "Subcutaneous Fat", 2: "Torso Fat",
                3: "Skeletal Muscle",  4: "Intermuscular Fat"}
legend_patches = [
    mpatches.Patch(facecolor=class_colors[v], label=f"{v} – {name}")
    for v, name in class_labels.items()
]
ax_lm.legend(
    handles=legend_patches,
    loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
    fontsize=9, framealpha=0.9,
)


fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
fig.savefig(OUT_PNG,  dpi=DPI, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
print(f"Saved: {OUT_PNG}")
