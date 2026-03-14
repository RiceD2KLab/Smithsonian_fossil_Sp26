"""
H5-backed YOLO dataset for Ultralytics YOLO26 training on palynomorph tiles.

This module streams the focus-stacked tiles directly from HDF5 files.

DATASET ASSUMPTIONS:
    - H5 files contain groups representing tiles (e.g., "tile_0_0", "tile_1_2")
    - Each tile group has:
        * focus_stacked: numpy array of shape (H, W, C) with RGB image data
        * bboxes: numpy array of shape (N, 4) with XYWH pixel coordinates
        * labels: numpy array of shape (N,) with integer class indices
    - Bounding boxes are validated (see is_valid_bounding_box function)
    - Images are converted from RGB to BGR (Ultralytics convention)

YOLO FORMAT:
    - Labels are stored as normalized center-xywh: (cx, cy, w, h) in [0, 1]
    - Class indices are 0-based integers
    - single_cls mode maps all classes to index 0 ("palynomorph")
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOCAL_RANK

from src.exploration.h5_utils import list_tile_jobs


H5_PREFIX: str = "h5::"

MINIMUM_BOX_AREA_PX: int = 16
"""Minimum bounding box area in pixels. Boxes smaller than this are filtered out."""

MINIMUM_ASPECT_RATIO: float = 0.25
"""Minimum aspect ratio (w/h or h/w). Boxes with extreme ratios are filtered out."""


def is_valid_bounding_box(
    box_width: int,
    box_height: int,
) -> bool:
    """
    Check if a bounding box meets validity criteria for training.

    Valid if:
        1. Width and height are both positive
        2. Area is at least MINIMUM_BOX_AREA_PX (16 pixels)
        3. Aspect ratio is not extreme (MINIMUM_ASPECT_RATIO and 1/MINIMUM_ASPECT_RATIO)

    Args:
        box_width: Width of the bounding box in pixels.
        box_height: Height of the bounding box in pixels.

    Returns:
        True if box passes all validity checks, False otherwise.
    """
    # Check for non-positive dimensions
    if box_width <= 0 or box_height <= 0:
        return False

    # Check minimum area requirement
    box_area: int = box_width * box_height
    if box_area < MINIMUM_BOX_AREA_PX:
        return False

    # Check aspect ratio (reject very thin/tall boxes)
    if box_height < MINIMUM_ASPECT_RATIO * box_width:
        return False
    if box_width < MINIMUM_ASPECT_RATIO * box_height:
        return False

    return True


# Bounding Box Coordinate Conversion

def convert_xywh_pixel_to_normalized_center(
    bboxes_xywh_pixel: np.ndarray,
    class_labels: np.ndarray,
    image_height: int,
    image_width: int,
    use_single_class: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel XYWH bounding boxes to YOLO normalized center-xywh format.

    Filters out invalid boxes and converts from XYWH pixel coordinates to normalized center-xywh.

    Args:
        bboxes_xywh_pixel: Array of shape (N, 4) with [x, y, width, height] in pixels.
        class_labels: Array of shape (N,) with integer class indices.
        image_height: Height of the source image in pixels.
        image_width: Width of the source image in pixels.
        use_single_class: If True, all labels are mapped to class 0.

    Returns:
        Tuple of (class_array, bbox_array):
            - class_array: Shape (M, 1) with class indices as float32
            - bbox_array: Shape (M, 4) with normalized [cx, cy, w, h] as float32
        Where M <= N after filtering invalid boxes.
    """
    valid_classes: list[float] = []
    valid_boxes: list[list[float]] = []

    # Iterate through each bounding box and its corresponding label
    for bbox_coords, label in zip(bboxes_xywh_pixel.tolist(), class_labels.tolist()):
        pixel_x, pixel_y, pixel_w, pixel_h = bbox_coords

        int_x, int_y = int(pixel_x), int(pixel_y)
        int_w, int_h = int(pixel_w), int(pixel_h)

        # Skip boxes that fail validation
        if not is_valid_bounding_box(int_w, int_h):
            continue

        # Convert top-left to center coordinates
        center_x: float = (pixel_x + pixel_w / 2.0) / image_width
        center_y: float = (pixel_y + pixel_h / 2.0) / image_height

        # Normalize dimensions by the image size
        normalized_width: float = pixel_w / image_width
        normalized_height: float = pixel_h / image_height

        valid_boxes.append([center_x, center_y, normalized_width, normalized_height])

        # Determine class index
        if use_single_class:
            class_index: float = 0.0
        else:
            class_index = float(int(label)) if label >= 0 else 0.0

        valid_classes.append(class_index)

    # Convert lists to numpy arrays with the correct shapes
    if valid_classes:
        class_array: np.ndarray = np.array(valid_classes, dtype=np.float32).reshape(-1, 1)
        bbox_array: np.ndarray = np.array(valid_boxes, dtype=np.float32)
    else:
        # Return empty arrays with the correct shapes for YOLO
        class_array = np.zeros((0, 1), dtype=np.float32)
        bbox_array = np.zeros((0, 4), dtype=np.float32)

    return class_array, bbox_array


def compute_cache_file_path(
    h5_root_directory: str,
    split_name: str,
    stems_hash: str,
    use_single_class: bool,
    use_best_plane: bool,
) -> Path:
    """
    Compute the path for the label cache file.

    Args:
        h5_root_directory: The directory containing the H5 files.
        split_name: The name of the dataset split (train, val, test).
        stems_hash: Hash of the sorted image stems for this split.
        use_single_class: Whether to use single-class mode.
        use_best_plane: Whether to use the best focal plane.

    Returns:
        The path to the cache file.
    """
    # Use suffix to distinguish single-class from multi-class caches
    class_mode_suffix: str = "_single" if use_single_class else "_multi"
    plane_mode_suffix: str = "_best" if use_best_plane else "_stacked"
    cache_filename: str = f".yolo_h5_{split_name}_{stems_hash[:12]}{class_mode_suffix}{plane_mode_suffix}.cache"

    return Path(h5_root_directory) / cache_filename


def build_labels_from_h5_files(
    image_file_identifiers: list[str],
    data_config: dict[str, Any],
    use_single_class: bool,
    use_best_plane: bool,
) -> list[dict[str, Any]]:
    """
    Build YOLO-format label dictionaries from the H5 tile files.

    Args:
        image_file_identifiers: List of H5 identifiers in format 'h5::/path::group'.
        data_config: The YOLO data configuration dictionary.
        use_single_class: Whether to use single-class mode.
        use_best_plane: Whether to use the best focal plane.

    Returns:
        A list of label dictionaries, one per valid tile. Each dictionary contains:
            - im_file: The original identifier string
            - shape: The shape of the image (height, width)
            - cls: The class indices (N, 1)
            - bboxes: The normalized boxes, shape (N, 4)
            - segments: An empty list (not used for detection)
            - keypoints: None (not used for detection)
            - normalized: True (boxes are in [0,1] range)
            - bbox_format: "xywh" (center-based format) (Ultralytics convention)
    """
    label_records: list[dict[str, Any]] = []

    for image_identifier in image_file_identifiers:
        # Validate we're dealing with an image from the H5 files
        if not image_identifier.startswith(H5_PREFIX):
            raise ValueError(
                f"Expected H5-backed image identifier starting with '{H5_PREFIX}', "
                f"got: {image_identifier}"
            )

        # "h5::/path/to/file.h5::group_name"
        _, h5_file_path, tile_group_name = image_identifier.split("::", 2)

        # Open H5 file and extract tile data
        with h5py.File(h5_file_path, "r") as h5_file:
            # Validate tile group and required datasets exist
            if tile_group_name not in h5_file:
                warnings.warn(
                    f"Tile group '{tile_group_name}' doesn't exist in {h5_file_path}."
                )
                continue
            
            # Check for the required dataset based on mode
            if use_best_plane:
                if "data" not in h5_file[tile_group_name]:
                    warnings.warn(
                        f"Tile group '{tile_group_name}' doesn't have a 'data' dataset for best_plane mode."
                    )
                    continue
                image_source = h5_file[tile_group_name]["data"]
            else:
                if "focus_stacked" not in h5_file[tile_group_name]:
                    warnings.warn(
                        f"Tile group '{tile_group_name}' doesn't have a 'focus_stacked' dataset."
                    )
                    continue
                image_source = h5_file[tile_group_name]["focus_stacked"]

            tile_group = h5_file[tile_group_name]

            # Extract bounding boxes and labels
            raw_bboxes: np.ndarray = np.array(tile_group["bboxes"])
            raw_labels: np.ndarray = np.array(tile_group["labels"])

            # Image dimensions from appropriate dataset
            # For "data" (H, W, C, Z), for "focus_stacked" (H, W, C)
            tile_height: int = int(image_source.shape[0])
            tile_width: int = int(image_source.shape[1])

        # Handle tiles with no bounding boxes or labels
        if raw_bboxes.size == 0 and raw_labels.size == 0:
            class_array = np.zeros((0, 1), dtype=np.float32)
            bbox_array = np.zeros((0, 4), dtype=np.float32)
        else:
            # Convert the bounding boxes to YOLO's expected input format
            class_array, bbox_array = convert_xywh_pixel_to_normalized_center(
                bboxes_xywh_pixel=raw_bboxes,
                class_labels=raw_labels,
                image_height=tile_height,
                image_width=tile_width,
                use_single_class=use_single_class,
            )

        # Label record conforming to Ultralytics expected format
        label_record: dict[str, Any] = {
            "im_file": image_identifier,
            "shape": (tile_height, tile_width),
            "cls": class_array,
            "bboxes": bbox_array,
            "segments": [],  # Not used in our use case
            "keypoints": None,  # Not used in our use case
            "normalized": True,  # Boxes are in [0,1] range
            "bbox_format": "xywh",  # Center-based xywh format
        }

        label_records.append(label_record)

    return label_records


class H5YOLODataset(YOLODataset):
    """
    Streams images and labels from H5 tile files.

    TExtends Ultralytics YOLODataset to load focus-stacked images
    (or a specific focal plane) directly from HDF5 files, to bypass 
    the standard image file approach (i.e., loading exported images).
    
    Tiles are selected by the given split from the `train_val_test.json` file.

    DATA LOADING:
        - Images come from 'focus_stacked' (or 'data' if use_best_plane=True) dataset in each H5 tile group
        - Images are converted from RGB (H5 storage) to BGR (Ultralytics convention)
        - Bounding boxes are read from the bboxes and labels datasets

    AUGMENTATION (handled by parent class, YOLODataset):
        - When augment=True: mosaic, mixup, hsv shifts, flips, scale, translate
        - When augment=False: only resize and letterbox padding
        - ANy additional specific augmentation hyperparameters can be passed in via the `hyp` argument

    CACHING:
        - Label metadata can be cached to avoid re-scanning the H5 files
        - Cache is keyed by (h5_root, split_name, image_stems_hash, use_single_class, use_best_plane)
    """

    def __init__(
        self,
        h5_root: str,
        splits_json_path: str,
        split_name: str,
        data: dict[str, Any],
        cache_labels: bool,
        single_cls: bool,
        use_best_plane: bool,
        imgsz: int,
        batch_size: int,
        augment: bool,
        hyp: Any,
        rect: bool,
        stride: int,
        pad: float,
        prefix: str,
        task: str,
        classes: list[int] | None,
        fraction: float,
        img_path: str | None = None,
        best_plane_root: str | None = None,
    ) -> None:
        """
        Initialize the H5-backed YOLO dataset.

        Args:
            h5_root: Directory containing .h5 tile files.
            splits_json_path: Path to train_val_test.json defining dataset splits.
            split_name: Which split to load ("train", "val", or "test").
            data: YOLO data configuration dict with 'names' and 'nc'.
            cache_labels: Whether to cache the label metadata to disk.
            single_cls: Whether to map all classes to a single "palynomorph" class.
            use_best_plane: Whether to use the best focal plane.
            imgsz: Target image size for training/validation.
            batch_size: Batch size.
            augment: Whether to enable training augmentations.
            hyp: Hyperparameter configuration (from Ultralytics args).
            rect: Whether to enable rectangular training (less padding).
            stride: Model stride for size calculations.
            pad: Padding factor for letterboxing.
            prefix: Logging prefix string (e.g., "train: ", "val: ", "test: ").
            task: Task type (should be "detect").
            classes: Optional list of class indices to filter.
            fraction: Fraction of data to use (1.0 = all tiles).
            img_path: Override image path (defaults to the H5 root directory).
        """
        # Store H5-specific configuration
        self._h5_root: str = os.path.abspath(h5_root)
        self._splits_json_path: str = os.path.abspath(splits_json_path)
        self._split_name: str = split_name
        self._cache_labels: bool = cache_labels
        self.single_cls: bool = single_cls
        self.use_best_plane: bool = use_best_plane
        self._best_plane_root: str | None = (
            os.path.abspath(best_plane_root) if best_plane_root else None
        )

        # Default img_path to the H5 root directory if not specified
        if img_path is None:
            img_path = self._h5_root

        # Disable built-in image caching; we load from the H5 files on demand
        # The 'cache' parameter controls Ultralytics RAM/disk image caching
        super().__init__(
            img_path=img_path,
            imgsz=imgsz,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache=None,  # Disable image caching for H5 streaming
            single_cls=single_cls,
            stride=stride,
            pad=pad,
            prefix=prefix,
            task=task,
            classes=classes,
            data=data,
            fraction=fraction,
        )

    def _resolve_h5_path(self, source_h5_path: str) -> tuple[str, bool]:
        """
        Resolve the H5 file path to read image data from.

        If best_plane_root is set and the corresponding cache file exists and is
        marked complete, return the cache path. Otherwise return the source path.
        """
        if self.use_best_plane and self._best_plane_root:
            filename = os.path.basename(source_h5_path)
            cache_path = os.path.join(self._best_plane_root, filename)
            if os.path.isfile(cache_path):
                try:
                    with h5py.File(cache_path, "r") as h5f:
                        if h5f.attrs.get("extraction_complete", False):
                            return cache_path, True
                except Exception:
                    # Cache file is corrupt or unreadable; fall back to source.
                    pass

        return source_h5_path, False

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """
        Build list of H5 tile identifiers for this split.

        Scans h5_root for all tile groups using list_tile_jobs, then filters
        to only include tiles whose source image stem is in the current split.

        Args:
            img_path: Path argument (unused, BUT NEEDED for YOLODataset API compatibility).

        Returns:
            List of identifier strings in the format "h5::/path/to/file.h5::group_name".
        """
        # Load split configuration
        with open(self._splits_json_path, "r", encoding="utf-8") as json_file:
            splits_data: dict[str, list[str]] = json.load(json_file)

        # Image stems belonging to this split
        split_image_stems: set[str] = set(splits_data[self._split_name])

        # Discover all tiles across all H5 files in h5_root
        all_tile_jobs: list[tuple[str, str, str]] = list_tile_jobs(self._h5_root)

        # Filter tiles to only those matching the current split
        # Each job is a tuple of (h5_path, image_stem, group_name)
        filtered_tiles: list[tuple[str, str, str]] = [
            (h5_path, image_stem, group_name)
            for h5_path, image_stem, group_name in all_tile_jobs
            if image_stem in split_image_stems
        ]

        tile_identifiers: list[str] = [
            f"{H5_PREFIX}{h5_path}::{group_name}"
            for h5_path, _, group_name in filtered_tiles
        ]

        return tile_identifiers

    def get_labels(self) -> list[dict[str, Any]]:
        """
        Build or load cached label metadata for all tiles in the current split.

        Checks for a valid cache file matching the current configuration.
        If not found, scans all the H5 files to extract the bounding boxes and labels.
        and caches the result.

        Returns:
            List of label dictionaries, one per tile in the current split.
        """
        # Load split configuration
        with open(self._splits_json_path, "r", encoding="utf-8") as json_file:
            splits_data: dict[str, list[str]] = json.load(json_file)

        # Compute hash of sorted image stems
        sorted_stems: list[str] = sorted(splits_data[self._split_name])
        stems_json: str = json.dumps(sorted_stems, sort_keys=True)
        stems_hash: str = hashlib.sha256(stems_json.encode()).hexdigest()

        cache_path: Path = compute_cache_file_path(
            h5_root_directory=self._h5_root,
            split_name=self._split_name,
            stems_hash=stems_hash,
            use_single_class=self.single_cls,
            use_best_plane=self.use_best_plane,
        )

        # Attempt to load from cache (only on the main process for distributed training)
        if self._cache_labels and cache_path.is_file() and LOCAL_RANK in (-1, 0):
            try:
                with open(cache_path, "rb") as cache_file:
                    cached_data: dict[str, Any] = pickle.load(cache_file)

                # Validate cache hash matches the current configuration
                if cached_data.get("hash") == stems_hash and "labels" in cached_data:
                    # Sync the im_files with the cached labels to prevent index mismatches
                    cached_labels: list[dict[str, Any]] = cached_data["labels"]
                    self.im_files = [label_dict["im_file"] for label_dict in cached_labels]
                    return cached_labels

            except (pickle.UnpicklingError, KeyError, EOFError):
                # Cache is corrupted or incompatible, so we just rebuild the labels
                pass

        # Build labels by scanning the H5 files
        computed_labels: list[dict[str, Any]] = build_labels_from_h5_files(
            image_file_identifiers=self.im_files,
            data_config=self.data,
            use_single_class=self.single_cls,
            use_best_plane=self.use_best_plane,
        )

        # Validate that we found some tiles in the current split
        if not computed_labels:
            raise RuntimeError(
                f"No valid tiles found for the current split '{self._split_name}' in {self._h5_root}. "
                "Verify that H5 groups contain 'focus_stacked', 'bboxes', and 'labels' datasets."
            )

        # Sync the im_files with the computed labels
        self.im_files = [label_dict["im_file"] for label_dict in computed_labels]

        # Save to cache (only on the main process)
        if self._cache_labels and LOCAL_RANK in (-1, 0):
            try:
                cache_data: dict[str, Any] = {
                    "hash": stems_hash,
                    "labels": computed_labels,
                }
                with open(cache_path, "wb") as cache_file:
                    pickle.dump(cache_data, cache_file)
            except (OSError, pickle.PicklingError):
                # Cache write failed but who cares
                pass

        return computed_labels

    def load_image(
        self,
        index: int,
        rect_mode: bool = True,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Load a single image from the H5 file by index.

        Reads either the focus_stacked dataset or the best focal plane
        from the corresponding H5 tile group and converts from RGB to 
        BGR (Ultralytics convention).

        Args:
            index: Index into the self.im_files for the image to load.
            rect_mode: Whether rectangular inference is enabled (unused here,
                        BUT NEEDED for YOLODataset API compatibility).

        Returns:
            Tuple of (image, original_shape, current_shape)
                - image: BGR numpy array of shape (H, W, C)
                - original_shape: (height, width) of the loaded image
                - current_shape: (height, width) after any transforms (same as the original shape for our use case)
        """
        image_identifier: str = self.im_files[index]

        # This dataset serves only H5-backed tiles; non-H5 identifiers are invalid
        if not image_identifier.startswith(H5_PREFIX):
            raise ValueError(
                f"Expected H5-backed image identifier (prefix '{H5_PREFIX}'), got: {image_identifier}"
            )

        # Parse H5 identifier
        _, h5_file_path, tile_group_name = image_identifier.split("::", 2)

        resolved_path, is_cache = self._resolve_h5_path(h5_file_path)

        # Load image from the resolved H5 file
        with h5py.File(resolved_path, "r") as h5_file:
            group = h5_file[tile_group_name]

            if is_cache:
                raw_image = np.array(group["best_plane"])
            elif self.use_best_plane:
                # Use focal_plane_ranking[0] to index into data (H, W, C, Z)
                best_z = int(np.array(group["focal_plane_ranking"])[0])
                raw_image: np.ndarray = np.array(group["data"][:, :, :, best_z])
            else:
                # Load precomputed focus-stacked image
                raw_image = np.array(group["focus_stacked"])

        # Convert RGB to BGR (Ultralytics convention)
        # The H5 have images in RGB from the original NDPI/microscopy format
        if raw_image.ndim == 3 and raw_image.shape[2] == 3:
            bgr_image: np.ndarray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = raw_image

        image_height: int = bgr_image.shape[0]
        image_width: int = bgr_image.shape[1]
        original_shape: tuple[int, int] = (image_height, image_width)

        # Update the augmentation buffer (used by mosaic/mixup augmentations)
        if self.augment:
            self.buffer.append(index)
            if len(self.buffer) > 1 and len(self.buffer) >= self.max_buffer_length:
                self.buffer.pop(0)

        # Return the image with shape information
        # Both shapes are the same since no resizing happens in the load_image method
        return bgr_image, original_shape, original_shape
