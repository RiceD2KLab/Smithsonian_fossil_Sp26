from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def _tile_groups(h5f: h5py.File) -> list[str]:
    """Return sorted list of tile group keys (tile_*)."""
    return sorted(k for k in h5f.keys() if k.startswith("tile_"))


def extract_single_file(
    source_path: str,
    output_path: str,
) -> tuple[str, int, str]:
    """
    Extract best focal plane for every tile in one H5 file.

    Args:
        source_path: Path to source H5 file (read-only).
        output_path: Path to write the cache H5 file.

    Returns:
        Tuple of (source_path, tiles_written, status_message).
    """
    if not os.path.isfile(source_path):
        return source_path, 0, "source file missing"

    # Skip if already complete
    if os.path.isfile(output_path):
        try:
            with h5py.File(output_path, "r") as h5f:
                if h5f.attrs.get("extraction_complete", False):
                    return source_path, 0, "skipped (already complete)"
        except Exception:
            # Corrupted or incomplete — overwrite below
            pass

    tiles_written = 0
    tmp_path = output_path + ".tmp"

    try:
        with h5py.File(source_path, "r") as src, h5py.File(tmp_path, "w") as dst:
            tile_keys = _tile_groups(src)

            for key in tile_keys:
                grp = src[key]

                # Validate required datasets exist
                if "data" not in grp or "focal_plane_ranking" not in grp:
                    # Older or partially processed tiles may not have focus metadata
                    continue

                focal_plane_ranking = np.array(grp["focal_plane_ranking"])
                if focal_plane_ranking.size == 0:
                    continue

                # Extract the best focal plane — pays gzip cost once per tile
                best_z = int(focal_plane_ranking[0])
                data_ds = grp["data"]

                if data_ds.ndim != 4:
                    continue

                # (H, W, C, Z) → select best_z
                best_plane = np.array(data_ds[:, :, :, best_z])

                bboxes = np.array(grp["bboxes"])
                labels = np.array(grp["labels"])

                out_grp = dst.create_group(key)
                out_grp.create_dataset(
                    "best_plane",
                    data=best_plane,
                    compression="lzf",
                    chunks=best_plane.shape,
                )
                out_grp.create_dataset("bboxes", data=bboxes, compression="lzf")
                out_grp.create_dataset("labels", data=labels, compression="lzf")

                # Copy tile-level attributes (x, y, w, h, num_annotations, etc.)
                for attr_key, attr_val in grp.attrs.items():
                    out_grp.attrs[attr_key] = attr_val

                tiles_written += 1

            # Copy file-level attributes
            for attr_key, attr_val in src.attrs.items():
                dst.attrs[attr_key] = attr_val

            # Mark as complete — used by resumability check and by dataset.py
            dst.attrs["extraction_complete"] = True

        # Atomic rename: only replace output if write fully succeeded
        os.replace(tmp_path, output_path)

    except Exception as exc:  # pragma: no cover - defensive clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return source_path, 0, f"FAILED: {exc}"

    return source_path, tiles_written, f"ok ({tiles_written} tiles)"


def run(h5_root: str, output_root: str, workers: int) -> None:
    """Extract best planes for all .h5 files under h5_root."""
    h5_root_path = Path(h5_root)
    if not h5_root_path.is_dir():
        raise FileNotFoundError(f"h5_root directory not found: {h5_root}")

    os.makedirs(output_root, exist_ok=True)

    h5_files = sorted(h5_root_path.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {h5_root}")

    print(f"Found {len(h5_files)} H5 files. Output root: {output_root}")
    print(f"Using {workers} worker processes.\n")

    jobs = [(str(p), os.path.join(output_root, p.name)) for p in h5_files]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_single_file, src, dst): src for src, dst in jobs
        }
        with tqdm(total=len(futures), unit="file") as pbar:
            for future in as_completed(futures):
                src_path, n_tiles, status = future.result()
                _ = n_tiles  # currently unused, but useful for debugging
                pbar.set_postfix_str(f"{Path(src_path).name}: {status}")
                pbar.update(1)

    print("\nBest-plane extraction complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract best focal planes from H5 tiles into a lightweight cache directory."
        ),
    )
    parser.add_argument(
        "--h5_root",
        type=str,
        required=True,
        help="Directory containing source .h5 tile files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory to write best-plane cache .h5 files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.h5_root, args.output_root, args.workers)


if __name__ == "__main__":
    main()

