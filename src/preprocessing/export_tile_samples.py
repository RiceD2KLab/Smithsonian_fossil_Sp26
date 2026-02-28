"""
Export a subset of tiles from generate_tiles H5 outputs to images
with annotations drawn, plus a manifest mapping each image to the original
slide and tile location for third-party verification.
"""

import argparse
import csv
import json
import os
import random

import h5py
import numpy as np
import cv2


def _load_metadata(tiles_dir: str) -> dict | None:
    path = os.path.join(tiles_dir, "metadata.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _index_to_name_map(metadata: dict | None) -> dict[int, str]:
    """Build label index -> category name from metadata.json label_map."""
    if not metadata or "label_map" not in metadata:
        return {}
    # label_map is category -> index; reverse to index -> category
    return {idx: name for name, idx in metadata["label_map"].items()}


def _tile_groups(h5f: h5py.File) -> list[str]:
    """Return sorted list of group names that are tiles (tile_*)."""
    return sorted(k for k in h5f.keys() if k.startswith("tile_") and isinstance(h5f[k], h5py.Group))


def _select_tile_keys(
    tile_keys: list[str],
    h5f: h5py.File,
    strategy: str,
    max_per_slide: int,
    rng: random.Random,
) -> list[str]:
    """
    Select up to max_per_slide tile group keys according to strategy.
    first_n: take the first N tiles in key order.
    with_annotations: take the first N tiles that have at least one annotation.
    random: shuffle and take N.
    """
    if strategy == "first_n":
        return tile_keys[:max_per_slide]
    if strategy == "with_annotations":
        with_ann = [k for k in tile_keys if h5f[k].attrs.get("num_annotations", 0) > 0]
        return with_ann[:max_per_slide]
    if strategy == "random":
        chosen = tile_keys.copy()
        rng.shuffle(chosen)
        return chosen[:max_per_slide]
    return tile_keys[:max_per_slide]


def _draw_annotations(img: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, index_to_name: dict[int, str]) -> np.ndarray:
    """Draw bboxes and label text on RGB image; returns BGR for cv2.imwrite."""
    out = img.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    for i in range(len(bboxes)):
        x, y, w, h = bboxes[i].astype(int)
        label_id = int(labels[i]) if i < len(labels) else -1
        label_name = index_to_name.get(label_id, str(label_id))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"{label_name} {i}", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _export_tile_image(
    group: h5py.Group,
    out_path: str,
    index_to_name: dict[int, str],
) -> None:
    """
    Load this tile's pixel data and annotations from the H5 group, draw boxes/labels, and save.
    Uses the 12th focal plane only (data[:, :, :, 12]). out_path already has the desired extension.
    !! From the exploration, on average, the 12th focal plane is the best.
    """
    data = group["data"][:]
    if data.ndim == 4:
        img = data[:, :, :, 12]
    else:
        img = data
    bboxes = group["bboxes"][:]
    labels = group["labels"][:]
    img_draw = _draw_annotations(img, bboxes, labels, index_to_name)
    cv2.imwrite(out_path, img_draw)


def run(
    tiles_dir: str,
    output_dir: str,
    max_per_slide: int = 10,
    max_slides: int = 5,
    strategy: str = "first_n",
    seed: int | None = None,
) -> None:
    """
    Export a subset of tiles from H5 files in tiles_dir to output_dir with
    images and manifest files.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Label index -> name for drawing; comes from generate_tiles metadata.json
    metadata = _load_metadata(tiles_dir)
    index_to_name = _index_to_name_map(metadata)
    if metadata:
        meta_out = os.path.join(output_dir, "metadata.json")
        with open(meta_out, "w") as f:
            json.dump(metadata, f, indent=2)

    rng = random.Random(seed)
    h5_files = sorted(f for f in os.listdir(tiles_dir) if f.endswith(".h5"))
    if max_slides is not None:
        h5_files = h5_files[:max_slides]

    manifest_entries: list[dict] = []
    ext = ".jpeg"

    for h5_name in h5_files:
        h5_path = os.path.join(tiles_dir, h5_name)
        image_name = os.path.splitext(h5_name)[0]
        slide_images_dir = os.path.join(images_dir, image_name)
        os.makedirs(slide_images_dir, exist_ok=True)

        with h5py.File(h5_path, "r") as h5f:
            source_ndpi = h5f.attrs.get("source_image", "")
            magnification = float(h5f.attrs.get("magnification", 0))
            tile_size = int(h5f.attrs.get("tile_size", 0))
            tile_keys = _tile_groups(h5f)
            selected = _select_tile_keys(tile_keys, h5f, strategy, max_per_slide, rng)

            for group_name in selected:
                grp = h5f[group_name]
                x_slide = int(grp.attrs["x"])
                y_slide = int(grp.attrs["y"])
                w = int(grp.attrs["w"])
                h = int(grp.attrs["h"])
                num_annotations = int(grp.attrs.get("num_annotations", 0))

                # Build per-annotation records for manifest (tile-local bbox + label name).
                bboxes = grp["bboxes"][:]
                labels = grp["labels"][:]
                annotations_list = []
                for i in range(len(bboxes)):
                    bbox = bboxes[i].tolist()
                    lid = int(labels[i]) if i < len(labels) else -1
                    annotations_list.append({
                        "bbox_tile": bbox,
                        "label_id": lid,
                        "label_name": index_to_name.get(lid, str(lid)),
                    })

                out_filename = f"tile_{x_slide}_{y_slide}{ext}"
                out_path = os.path.join(slide_images_dir, out_filename)
                _export_tile_image(grp, out_path, index_to_name)

                # One manifest row per exported tile: slide path, tile position, and annotation list.
                rel_image_path = os.path.join("images", image_name, out_filename)
                entry = {
                    "source_ndpi": source_ndpi,
                    "source_h5": h5_path,
                    "tile_group": group_name,
                    "x_slide": x_slide,
                    "y_slide": y_slide,
                    "w": w,
                    "h": h,
                    "exported_image": rel_image_path,
                    "magnification": magnification,
                    "tile_size": tile_size,
                    "num_annotations": num_annotations,
                    "annotations": annotations_list,
                }
                manifest_entries.append(entry)

    # Full manifest: one object per tile with nested annotations list (for tooling).
    manifest_json_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_json_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)

    # Flat CSV of the same tiles (no nested annotations) for spreadsheet review.
    csv_path = os.path.join(output_dir, "manifest.csv")
    if manifest_entries:
        fieldnames = [
            "source_ndpi", "source_h5", "tile_group", "x_slide", "y_slide", "w", "h",
            "exported_image", "magnification", "tile_size", "num_annotations",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for e in manifest_entries:
                row = {k: e[k] for k in fieldnames if k in e}
                writer.writerow(row)

    # README.txt
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(
            "Exported tile samples for 1:1 verification. "
            "manifest.csv and manifest.json map each image to the original slide path and tile position (x_slide, y_slide). "
            "Images show the 12th focal plane only; boxes and labels are the annotations from the H5 tiles.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a subset of tiles from generate_tiles H5 output to images and a manifest for verification."
    )
    parser.add_argument(
        "tiles_dir",
        help="Directory containing H5 files (and optionally metadata.json) from generate_tiles.py",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write exported images, manifest.json, manifest.csv, and README.txt",
    )
    parser.add_argument(
        "--max-per-slide",
        type=int,
        default=10,
        help="Max number of tiles to export per slide (default: 10)",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=5,
        help="Max number of slides (H5 files) to process (default: 5)",
    )
    parser.add_argument(
        "--strategy",
        choices=["first_n", "random", "with_annotations"],
        default="first_n",
        help="How to select tiles per slide: first_n, random, or with_annotations (default: first_n)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for strategy=random",
    )
    args = parser.parse_args()
    run(
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        max_per_slide=args.max_per_slide,
        max_slides=args.max_slides,
        strategy=args.strategy,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
