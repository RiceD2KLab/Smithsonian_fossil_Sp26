import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.postprocess_tiles import maximum_intensity_projection

def test_mip_on_sample(h5_path, output_dir, max_tiles=20):
    """
    Test function: Run MIP on up to max_tiles from a single h5 file,
    save the resulting images to output_dir using matplotlib.
    """
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(h5_path, 'r') as h5f:
        for i, tile_name in enumerate(list(h5f.keys())[:max_tiles]):
            group = h5f[tile_name]
            if not isinstance(group, h5py.Group) or 'data' not in group.keys():
                continue
            data = group['data'][:]
            if not hasattr(data, 'ndim') or data.ndim != 4:
                continue
            mip = maximum_intensity_projection(data)
            img = mip
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            plt.figure(figsize=(6, 6))
            if img.shape[-1] == 1:
                plt.imshow(img[..., 0], cmap='gray')
            else:
                plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{tile_name}_mip.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

if __name__ == "__main__":
    TEST_H5_PATH = "/path/to/test_file.h5"  # Update with actual test file path
    TEST_OUTPUT_DIR = "/path/to/output/dir"  # Update with desired output directory
    test_mip_on_sample(TEST_H5_PATH, TEST_OUTPUT_DIR)
