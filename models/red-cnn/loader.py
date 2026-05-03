import os
from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


_TISSUES = ['subcutaneous_fat', 'torso_fat', 'skeletal_muscle', 'intermuscular_fat']
# label values 1-4 in that order; 0 = background


class ct_dataset(Dataset):
    """Loads paired LDCT/NDCT PNGs from png_dataset/{mode}/1mm/{QD,FD}/.
    Optionally loads and combines 4-tissue binary masks into a single-channel
    label map normalised to [0, 1].

    Training mode (patch_size set): __getitem__ returns
        (ldct_patches, ndct_patches)                    # use_mask=False
        (ldct_patches, ndct_patches, mask_patches)      # use_mask=True
    where each array is (patch_n, H, W).

    Val/test mode (patch_size=None): __getitem__ returns
        (ldct, ndct, fname)                             # use_mask=False
        (ldct, ndct, mask, fname)                       # use_mask=True
    where ldct/ndct/mask are (H, W) float32 arrays.

    PNG pixel values are expected to be raw HU values (float or uint16).
    They are clipped to [norm_range_min, norm_range_max] and normalised to [0,1].
    """

    def __init__(self, mode, data_path, mask_dir=None,
                 norm_range_min=-1024.0, norm_range_max=3072.0,
                 patch_n=None, patch_size=None):
        assert mode in ('train', 'val', 'test'), "mode must be 'train', 'val', or 'test'"

        self.mode = mode
        self.mask_dir = mask_dir
        self.norm_min = norm_range_min
        self.norm_max = norm_range_max
        self.patch_n = patch_n
        self.patch_size = patch_size

        qd_dir = os.path.join(data_path, mode, '1mm', 'QD')
        fd_dir = os.path.join(data_path, mode, '1mm', 'FD')

        qd_files = sorted(glob(os.path.join(qd_dir, '*.png')))
        ndct_map = {os.path.basename(f): f for f in sorted(glob(os.path.join(fd_dir, '*.png')))}

        self.filenames = []
        self.ldct_files = []
        self.ndct_files = []
        for qd_path in qd_files:
            qd_name = os.path.basename(qd_path)
            fd_name = qd_name.replace('_QD_', '_FD_')
            if fd_name in ndct_map:
                self.filenames.append(qd_name)
                self.ldct_files.append(qd_path)
                self.ndct_files.append(ndct_map[fd_name])

        if len(self.filenames) == 0:
            raise FileNotFoundError(
                f"No paired QD/FD PNGs found in {qd_dir} and {fd_dir}")

    def _load_png(self, path):
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(arr, self.norm_min, self.norm_max)
        arr = (arr - self.norm_min) / (self.norm_max - self.norm_min)
        return arr

    def _load_mask(self, filename):
        label_map = np.zeros((512, 512), dtype=np.float32)
        for i, tissue in enumerate(_TISSUES):
            path = os.path.join(self.mask_dir, tissue, self.mode, '1mm', 'QD', filename)
            if os.path.exists(path):
                mask = np.array(Image.open(path).convert('L'), dtype=np.float32)
                label_map[mask > 0] = float(i + 1)
        return label_map / 4.0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        ldct = self._load_png(self.ldct_files[idx])
        ndct = self._load_png(self.ndct_files[idx])
        mask = self._load_mask(fname) if self.mask_dir is not None else None

        if self.patch_size and self.patch_n:
            return _extract_patches(ldct, ndct, mask, self.patch_n, self.patch_size)
        else:
            if mask is not None:
                return ldct, ndct, mask, fname
            return ldct, ndct, fname


def _extract_patches(ldct, ndct, mask, patch_n, patch_size):
    h, w = ldct.shape
    ldct_p, ndct_p, mask_p = [], [], []
    for _ in range(patch_n):
        top = np.random.randint(0, h - patch_size)
        left = np.random.randint(0, w - patch_size)
        s = (slice(top, top + patch_size), slice(left, left + patch_size))
        ldct_p.append(ldct[s])
        ndct_p.append(ndct[s])
        if mask is not None:
            mask_p.append(mask[s])
    if mask is not None:
        return np.array(ldct_p), np.array(ndct_p), np.array(mask_p)
    return np.array(ldct_p), np.array(ndct_p)


def get_loader(mode, data_path, mask_dir=None,
               norm_range_min=-1024.0, norm_range_max=3072.0,
               patch_n=None, patch_size=None,
               batch_size=16, num_workers=6):
    dataset = ct_dataset(
        mode=mode,
        data_path=data_path,
        mask_dir=mask_dir,
        norm_range_min=norm_range_min,
        norm_range_max=norm_range_max,
        patch_n=patch_n,
        patch_size=patch_size,
    )
    shuffle = (mode == 'train')
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)
