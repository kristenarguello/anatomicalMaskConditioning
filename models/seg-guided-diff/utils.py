from pathlib import Path
from typing_extensions import Literal
from PIL import Image


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def get_all_images_paths(img_dir: str, dose: Literal["QD", "FD"]) -> list[str]:
    """
    Returns a list of all 1mm images paths.
    Args:
        img_dir (str): Path to the image directory.
        dose (Literal["QD", "FD"]): Dose type, either "QD" (quarter dose) or "FD" (full dose).
            FD = for the target images
            QD = low dose images, used for conditioning
    """
    mms = ["1mm"]
    all_images = []
    for mm in mms:
        dose_dir = Path(img_dir) / mm / dose
        if not dose_dir.is_dir():
            print(f"Warning: Directory not found: {dose_dir}")
            continue
        # get all images inside dose_dir
        images = sorted(dose_dir.glob("*.png"))
        all_images.extend([str(img_path) for img_path in images])
    return all_images
