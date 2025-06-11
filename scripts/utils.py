import os
from PIL import Image

def crop_and_resize(img, size):
    min_dim = min(img.size)
    img = img.crop(
        (
            (img.width - min_dim) // 2,
            (img.height - min_dim) // 2,
            (img.width + min_dim) // 2,
            (img.height + min_dim) // 2,
        )
    )
    return img.resize((size, size), resample=Image.BILINEAR)


def create_missing_dirs(path):
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir)
        except Exception:
            raise OSError(f"Failed to create directory {parent_dir}")
    return path
