import os
import numpy as np
import nibabel as nib
from PIL import Image

# Class ID mapping for TotalSegmentator tissue_4_types task
CLASS_MAP = {
    1: "subcutaneous_fat",
    2: "torso_fat",  # visceral fat
    3: "skeletal_muscle",
    4: "intermuscular_fat",
}


def save_mask(mask_bool, out_path):
    # Save as 8-bit grayscale PNG: 0 (background) or 255 (class)
    Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L").save(out_path)


def export_multilabel_nii_to_pngs(nii_path, out_root, split, basename_prefix):
    """
    Convert a multilabel NIfTI volume to per-class binary PNG masks.

    nii_path: path to the multilabel NIfTI file (.nii or .nii.gz)
    out_root: root directory for output masks (seg_dir)
    split: 'train' | 'val' | 'test'
    basename_prefix: filename prefix for output files (e.g. 'L031')
    """
    vol = nib.load(nii_path)  # type: ignore
    data = np.asarray(vol.get_fdata()).astype(np.int32)  # type: ignore
    assert data.ndim == 3, "Expected 3D volume"

    Z = data.shape[2]
    for z in range(Z):
        label2d = data[:, :, z]
        for lab, name in CLASS_MAP.items():
            mask = label2d == lab
            out_dir = os.path.join(out_root, name, split)
            os.makedirs(out_dir, exist_ok=True)
            fn = f"{basename_prefix}_z{z:03d}.png"
            save_mask(mask, os.path.join(out_dir, fn))
