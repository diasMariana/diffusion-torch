import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path

sys.path.append("..")
from utils import create_missing_dirs, crop_and_resize

if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_dir", help="Path to dataset directory")
    parser.add_argument("--extension", help="Image format", default="jpg", type=str)
    parser.add_argument("--size", help="Image format", default="64", type=int)
    parser.add_argument("output_dir", help="Path to dataset directory")
    args, extra_args_list = parser.parse_known_args()

    # Resize and save
    print("Resizing data...")
    for fname in tqdm(Path(args.data_dir).rglob(f"*{args.extension}")):
        im_name = fname.name
        extension = im_name.split(".")[-1]
        if extension == args.extension:
            im = Image.open(fname)
            im = crop_and_resize(im, args.size)
            im.save(create_missing_dirs(os.path.join(args.output_dir, im_name)))
    print("Done!")
