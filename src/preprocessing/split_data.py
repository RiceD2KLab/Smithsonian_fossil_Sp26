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
    import argparse
    
    parser = argparse.ArgumentParser(description="Split data into train, val, and test sets.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input h5 files directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory to save splits.")
    parser.add_argument("--seed", type=int, default=67, help="Random seed for splitting.")
    
    args = parser.parse_args()

    splits = split_data(args.input_dir, seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save splits into separate text files
    for split_name, files in splits.items():
        if split_name == "train":
            filename = "training_set.txt"
        elif split_name == "val":
            filename = "val_set.txt"
        elif split_name == "test":
            filename = "test_set.txt"
        else:
            filename = f"{split_name}_set.txt"
            
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "w") as f:
            for file_name in files:
                f.write(f"{file_name}\n")
    
    print(f"Saved splits to {args.output_dir}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    