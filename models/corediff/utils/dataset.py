import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from natsort import natsorted

# Labels for the unified mask
CLASS_MAP = {
    "subcutaneous_fat":  1,
    "torso_fat":         2,
    "skeletal_muscle":   3,
    "intermuscular_fat": 4,
}

class CTDataset(Dataset):
    """
    Drop-in replacement that loads matched PNG images.

    img_root/{split}/1mm/QD/*.png   — quarter-dose LDCT
    img_root/{split}/1mm/FD/*.png   — full-dose NDCT
    mask_root/{class}/{split}/1mm/QD/*.png — per-class binary masks
    """

    def __init__(self, img_root, split, context=True, mask_dir=None, **kwargs):
        # kwargs swallows unused original args (dataset, test_id, dose)
        self.context = context
        self.use_mask = mask_dir is not None

        qd_dir = osp.join(img_root, split, "1mm", "QD")
        fd_dir = osp.join(img_root, split, "1mm", "FD")

        self.qd_paths = natsorted(glob(osp.join(qd_dir, "*.png")))
        self.fd_paths = natsorted(glob(osp.join(fd_dir, "*.png")))

        assert len(self.qd_paths) == len(self.fd_paths), \
            f"QD/FD count mismatch: {len(self.qd_paths)} vs {len(self.fd_paths)}"
        assert len(self.qd_paths) > 0, f"No images found in {qd_dir}"

        # Context mode: skip first and last slices (no neighbors)
        if context:
            self.valid_indices = list(range(1, len(self.qd_paths) - 1))
        else:
            self.valid_indices = list(range(len(self.qd_paths)))

        # Build per-class mask paths
        self.class_png_paths = {}
        if self.use_mask:
            assert mask_dir is not None, "mask_dir must be provided if use_mask is True"
            for class_name in CLASS_MAP:
                cdir = osp.join(mask_dir, class_name, split, "1mm", "QD")
                if not osp.isdir(cdir):
                    print(f"  WARNING: mask dir not found: {cdir}")
                    continue
                cpaths = natsorted(glob(osp.join(cdir, "*.png")))
                assert len(cpaths) == len(self.qd_paths), \
                    f"Mask count mismatch for {class_name}: {len(cpaths)} vs {len(self.qd_paths)}"
                self.class_png_paths[class_name] = cpaths

        print(f"CTDataset [{split}]: {len(self.valid_indices)} slices "
              f"(context={context}, masks={'yes' if self.use_mask else 'no'})")

    def _load_gray(self, path):
        """Load grayscale PNG as float32 [0, 1], shape (H, W)."""
        return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0

    def __getitem__(self, index):
        idx = self.valid_indices[index]

        # Target: full-dose NDCT
        target = self._load_gray(self.fd_paths[idx])[np.newaxis, ...]  # (1,H,W)

        # Input: quarter-dose LDCT
        if self.context:
            inp = np.stack([
                self._load_gray(self.qd_paths[idx + offset])
                for offset in [-1, 0, 1]
            ], axis=0)  # (3,H,W)
        else:
            inp = self._load_gray(self.qd_paths[idx])[np.newaxis, ...]  # (1,H,W)

        # Mask: merge per-class binary PNGs into single label map
        if self.use_mask and self.class_png_paths:
            h, w = target.shape[1], target.shape[2]
            label_map = np.zeros((h, w), dtype=np.float32)
            for class_name, cpaths in self.class_png_paths.items():
                binary = np.array(Image.open(cpaths[idx]).convert("L"))
                label_map[binary > 0] = CLASS_MAP[class_name]
            # Normalize to [0, 1]
            label_map = label_map / max(CLASS_MAP.values())
            mask = label_map[np.newaxis, ...]  # (1,H,W)
            return inp, target, mask

        return inp, target

    def __len__(self):
        return len(self.valid_indices)
