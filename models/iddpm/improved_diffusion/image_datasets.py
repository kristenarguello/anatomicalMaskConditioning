from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


# ---------------------------------------------------------------------------
# CT image translation dataset (LDCT -> NDCT, optional segmentation mask)
# ---------------------------------------------------------------------------

import os as _os

# Tissue class names in label order: class index i+1 corresponds to MASK_CLASS_NAMES[i].
#   label 0 = background
#   label 1 = subcutaneous_fat
#   label 2 = torso_fat
#   label 3 = skeletal_muscle
#   label 4 = intermuscular_fat
MASK_CLASS_NAMES = [
    "subcutaneous_fat",
    "torso_fat",
    "skeletal_muscle",
    "intermuscular_fat",
]


def load_ct_data(
    *,
    data_dir,
    batch_size,
    image_size,
    use_mask=False,
    mask_root="",
    deterministic=False,
):
    """
    Generator over (ndct, kwargs) pairs for paired CT denoising.

    *data_dir* may be a single path **or a comma-separated list of paths**,
    e.g. ``"/data/train,/data/val"``.  File lists from all directories are
    pooled so the dataset sees all splits at once.

    CT layout under each *data_dir* entry::

        1mm/QD/*.png   - low-dose CT (LDCT)
        1mm/FD/*.png   - full-dose CT (NDCT, training target)

    Mask layout (only needed when use_mask=True)::

        {mask_root}/{class_name}/{split}/1mm/QD/*.png

    The split name is derived automatically from each file's path, so mixing
    splits (e.g. train + val) works without any extra configuration.

    Yields batches of (ndct [B,1,H,W], dict) where the dict contains:
        ``ldct``  - [B,1,H,W] float32 in [-1, 1]
        ``mask``  - [B,1,H,W] float32 in [0, 1]  (only when use_mask=True)
    """
    if use_mask:
        assert mask_root, "--mask_root must be set when --use_mask True"

    # Support comma-separated list of directories
    data_dirs = [d.strip() for d in data_dir.split(",") if d.strip()]

    all_ldct_paths = []
    all_ndct_paths = []

    for d in data_dirs:
        ldct_dir = bf.join(d, "1mm", "QD")
        ndct_dir = bf.join(d, "1mm", "FD")

        ldct_names = sorted(
            f for f in bf.listdir(ldct_dir) if f.lower().endswith(".png")
        )
        ndct_names = sorted(
            f for f in bf.listdir(ndct_dir) if f.lower().endswith(".png")
        )
        assert len(ldct_names) == len(ndct_names), (
            f"LDCT/NDCT file lists differ in {d}: "
            f"{len(ldct_names)} vs {len(ndct_names)}"
        )

        all_ldct_paths.extend(bf.join(ldct_dir, f) for f in ldct_names)
        all_ndct_paths.extend(bf.join(ndct_dir, f) for f in ndct_names)

    dataset = CTDataset(
        image_size,
        all_ldct_paths,
        all_ndct_paths,
        mask_root=mask_root if use_mask else None,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )
    while True:
        yield from loader


class CTDataset(Dataset):
    """
    Paired LDCT / NDCT (/ mask) dataset.

    LDCT and NDCT are single-channel grayscale PNGs (8-bit or 16-bit).
    Normalised to [-1, 1]:
        uint16  ->  arr / 32767.5 - 1.0
        uint8   ->  arr / 127.5   - 1.0

    Masks are stored as four separate binary PNGs (one per tissue class).
    They are merged into a single multi-class label map [1, H, W] in [0, 1]:
        pixel value = class_label / 4,  where class_label ∈ {0,1,2,3,4}
        0 = background
        1 = subcutaneous_fat  (MASK_CLASS_NAMES[0])
        2 = torso_fat         (MASK_CLASS_NAMES[1])
        3 = skeletal_muscle   (MASK_CLASS_NAMES[2])
        4 = intermuscular_fat (MASK_CLASS_NAMES[3])

    The mask split is derived from each file's own path
    (.../split/1mm/QD/file.png), so pooling multiple splits works correctly.

    Returns (ndct_array [1,H,W], out_dict) where out_dict always contains
    ``ldct`` and optionally ``mask``.
    """

    def __init__(
        self,
        resolution,
        ldct_paths,
        ndct_paths,
        mask_root=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_ldct = ldct_paths[shard:][::num_shards]
        self.local_ndct = ndct_paths[shard:][::num_shards]
        self.mask_root = mask_root  # None or path string

    def __len__(self):
        return len(self.local_ldct)

    def __getitem__(self, idx):
        ldct_path = self.local_ldct[idx]
        ldct = self._load_ct_image(ldct_path)
        ndct = self._load_ct_image(self.local_ndct[idx])
        out_dict = {"ldct": ldct}
        if self.mask_root is not None:
            out_dict["mask"] = self._load_combined_mask(ldct_path)
        return ndct, out_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_ct_image(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        if pil_image.size != (self.resolution, self.resolution):
            pil_image = pil_image.resize(
                (self.resolution, self.resolution), resample=Image.BICUBIC
            )

        arr = np.array(pil_image)

        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 32767.5 - 1.0
        else:
            arr = arr.astype(np.float32) / 127.5 - 1.0

        if arr.ndim == 2:
            arr = arr[np.newaxis]
        else:
            arr = arr[:1]

        return arr

    def _load_combined_mask(self, ldct_path):
        """
        Load 4 binary mask PNGs for *ldct_path* and merge via overlay.

        The split name is derived from the file's own path:
            .../split/1mm/QD/file.png  ->  split = parents[2].name

        Returns float32 array [1, H, W] with values in [0, 1].
        """
        from pathlib import Path as _Path
        split = _Path(ldct_path).parents[2].name
        filename = _os.path.basename(ldct_path)

        combined = None
        for class_idx, cls in enumerate(MASK_CLASS_NAMES):
            mask_path = bf.join(self.mask_root, cls, split, "1mm", "QD", filename)
            with bf.BlobFile(mask_path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()

            if pil_image.size != (self.resolution, self.resolution):
                pil_image = pil_image.resize(
                    (self.resolution, self.resolution), resample=Image.NEAREST
                )

            arr = np.array(pil_image)
            if arr.ndim == 3:
                arr = arr[:, :, 0]

            if combined is None:
                combined = np.zeros(arr.shape, dtype=np.float32)
            combined[arr > 0] = float(class_idx + 1)

        combined = combined / 4.0
        return combined[np.newaxis]  # [1, H, W]
