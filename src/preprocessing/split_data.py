import os
import json
import random
from src.exploration.h5_utils import list_h5_paths

def split_data(image_dir, seed=None):

    # Read all processed h5 files and extract image names (without .h5 extension)
    h5_files = list_h5_paths(image_dir)
    if not h5_files:
        raise ValueError(f"No files found in {image_dir}")
    image_names = [os.path.splitext(os.path.basename(f))[0].strip(".h5") for f in h5_files]

    # Calculate number of samples for train/val/test splits
    n = len(image_names)
    n_val = int(n * 0.15)
    n_test = int(n * 0.15)
    n_train = n - n_val - n_test

    # Randomly shuffle and split into train/val/test (70/15/15)
    if seed is not None:
        random.seed(seed)
    random.shuffle(image_names)
    train = image_names[:n_train]
    val = image_names[n_train:n_train+n_val]
    test = image_names[n_train+n_val:]

    return {"train": train, "val": val, "test": test}

if __name__ == "__main__":
    INPUT_DIR = "/path/to/h5/files"
    OUTPUT_PATH = "/path/to/output/train_val_test.json"
    splits = split_data(INPUT_DIR, seed=42)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    
    print(splits)
    print(len(splits["train"]), len(splits["val"]), len(splits["test"]))
    