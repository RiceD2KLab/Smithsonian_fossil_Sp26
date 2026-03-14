"""
dataset.py — PyTorch Dataset for H5-backed palynomorph tile data.

Each .h5 file corresponds to one slide image and contains tile groups named
``tile_<x>_<y>``. Every group stores five datasets:

    data                (H, W, C, Z)  uint8  — RGB image stack across focal planes
    bboxes              (N, 4)        float  — bounding boxes in XYWH pixel format
    labels              (N,)          int    — integer class labels (1-indexed)
    focal_plane_ranking (Z,)          int64  — Z indices ranked sharpest-first
    focus_stacked       (H, W, C)     uint8  — precomputed focus-stacked RGB image

The dataset supports two image modes (see ``ImageMode``):
    - BEST_PLANE:   reads ``focal_plane_ranking[0]`` to select the sharpest Z plane
                    from ``data`` without any manual sharpness computation.
    - FOCUS_STACK:  reads the precomputed ``focus_stacked`` image directly.

Bboxes are converted from XYWH (H5 storage format) to XYXY (torchvision format),
and degenerate boxes (area < 16 px² or extreme aspect ratio) are filtered out.

Dependencies:
    h5py, numpy, torch, torchvision, Pillow

Typical usage:
    from torchvision import transforms
    from dataset import TileDataset, ImageMode

    transform = transforms.Compose([transforms.ToTensor()])

    # Use the precomputed best focal plane
    dataset = TileDataset(h5_root="data/tiles", mode=ImageMode.BEST_PLANE, transform=transform)

    # Use the precomputed focus-stacked image
    dataset = TileDataset(h5_root="data/tiles", mode=ImageMode.FOCUS_STACK, transform=transform)

    image, target = dataset[0]
"""

import os
from enum import Enum
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.exploration.h5_utils import list_tile_jobs


# Image mode selection

class ImageMode(Enum):
    """
    Controls which precomputed image is loaded from the H5 file per tile.

    Attributes:
        BEST_PLANE:  Reads ``focal_plane_ranking[0]`` to index into ``data``
                     (H, W, C, Z), returning the single sharpest focal plane
                     as an (H, W, C) image.
        FOCUS_STACK: Reads the precomputed ``focus_stacked`` dataset directly,
                     returning a single (H, W, C) image that combines sharpness
                     across all focal planes.
    """

    BEST_PLANE = "best_plane"
    FOCUS_STACK = "focus_stack"


# Dataset

class TileDataset(Dataset):
    """
    PyTorch Dataset that loads palynomorph tile images from H5 files.

    At construction time the dataset calls ``h5_utils.list_tile_jobs`` to
    build an index of all tile groups found across every H5 file. Each
    ``__getitem__`` call opens the relevant H5 group, reads the appropriate
    precomputed image according to ``mode``, and returns a (image, target)
    pair ready for use with torchvision Faster R-CNN / DETR style detection
    pipelines.

    Args:
        h5_root:   Path to a directory of ``.h5`` files, or to a single ``.h5`` file.
        mode:      ``ImageMode.BEST_PLANE`` to load the sharpest focal plane via
                   ``focal_plane_ranking``, or ``ImageMode.FOCUS_STACK`` to load
                   the precomputed focus-stacked image. Defaults to ``ImageMode.BEST_PLANE``.
        transform: Optional callable (e.g. ``torchvision.transforms.ToTensor()``)
                   applied to the PIL image before it is returned.

    Attributes:
        tiles: List of ``(h5_path, image_stem, group_name)`` tuples — one entry
               per tile discovered during initialisation.
    """

    def __init__(
        self,
        h5_root: str,
        mode: ImageMode = ImageMode.BEST_PLANE,
        transform: Optional[Callable] = None,
    ):
        self.h5_root = h5_root
        self.mode = mode
        self.transform = transform

        # Delegate file discovery and tile indexing to the shared utility
        self.tiles: list[tuple[str, str, str]] = list_tile_jobs(h5_root)


    # Internal helpers

    def _load_frame(self, group: h5py.Group) -> np.ndarray:
        """
        Load a single (H, W, C) uint8 image frame from an open H5 tile group.

        Selects the image according to ``self.mode``:
            - BEST_PLANE:  indexes ``data`` at ``focal_plane_ranking[0]``.
            - FOCUS_STACK: reads ``focus_stacked`` directly.

        Args:
            group: Open ``h5py.Group`` for the tile (e.g. ``h5f["tile_x_y"]``).

        Returns:
            numpy array of shape (H, W, C) and dtype uint8.
        """
        if self.mode == ImageMode.FOCUS_STACK:
            return np.array(group["focus_stacked"])

        # BEST_PLANE: use the first entry of the precomputed ranking
        best_z = int(np.array(group["focal_plane_ranking"])[0])
        return np.array(group["data"])[:, :, :, best_z]

    def _build_target(
        self,
        bboxes_xywh: np.ndarray,
        labels_raw: np.ndarray,
        idx: int,
    ) -> dict:
        """
        Build the detection target dictionary for one tile.

        Converts bounding boxes from XYWH (H5 storage format) to XYXY
        (torchvision format), discards degenerate boxes (area < 16 px² or
        extreme aspect ratio), and packages everything into tensors.

        Args:
            bboxes_xywh: Float array of shape (N, 4) with columns [x, y, w, h]
                         in pixel coordinates.
            labels_raw:  Integer array of shape (N,) with 1-indexed class labels.
            idx:         Dataset index of the current tile, stored in ``image_id``.

        Returns:
            Dict with keys:
                boxes    (FloatTensor [M, 4]) — XYXY boxes for the M valid annotations.
                labels   (Int64Tensor [M])    — class label per box.
                image_id (Int64Tensor [1])    — dataset index of this tile.
                area     (FloatTensor [M])    — pixel area of each box.
                iscrowd  (Int64Tensor [M])    — all zeros (required by COCO evaluator).
        """
        boxes, labels = [], []

        if bboxes_xywh.size > 0 and labels_raw.size > 0:
            for (x, y, w, h), label in zip(bboxes_xywh.tolist(), labels_raw.tolist()):
                x, y, w, h = int(x), int(y), int(w), int(h)
                if w > 0 and h > 0 and w * h >= 16 and not (h <= 0.25 * w or w <= 0.25 * h):  # check for aspect ratio to catch cropped annotation boxes
                    boxes.append([x, y, x + w, y + h])  # XYWH → XYXY
                    labels.append(int(label))

        target_boxes = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        target_labels = (
            torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64)
        )
        area = (target_boxes[:, 3] - target_boxes[:, 1]) * (target_boxes[:, 2] - target_boxes[:, 0])

        return {
            "boxes": target_boxes,
            "labels": target_labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((target_boxes.shape[0],), dtype=torch.int64),
        }


    # PyTorch Dataset protocol

    def __len__(self) -> int:
        """
        Return the total number of tiles across all H5 files.

        Returns:
            Integer count of tiles in the dataset.
        """
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple:
        """
        Load and return the image–target pair for the tile at position ``idx``.

        Opens the corresponding H5 group, loads the image via ``_load_frame``
        according to ``self.mode``, converts it to a PIL Image, applies any
        configured transform, and builds the detection target dict.

        Args:
            idx: Integer index in [0, len(dataset)).

        Returns:
            Tuple of:
                image  — PIL Image (or transformed tensor) of shape (C, H, W).
                target — Dict as described in ``_build_target``.
        """
        h5_path, _image_stem, group_name = self.tiles[idx]

        with h5py.File(h5_path, "r") as h5f:
            group = h5f[group_name]
            frame = self._load_frame(group)             # (H, W, C)
            bboxes_xywh = np.array(group["bboxes"])     # (N, 4) XYWH
            labels_raw = np.array(group["labels"])      # (N,)

        image = Image.fromarray(frame.astype(np.uint8), mode="RGB")

        if self.transform:
            image = self.transform(image)

        target = self._build_target(bboxes_xywh, labels_raw, idx)
        return image, target


    # Public utility methods

    def get_index_by_tile(self, image_stem: str, group_name: str) -> Optional[int]:
        """
        Look up the dataset index for a tile by its (image_stem, group_name) identity.

        Args:
            image_stem: Slide filename without extension, e.g. ``"slide01"``.
            group_name: H5 group key for the tile, e.g. ``"tile_3_7"``.

        Returns:
            Integer dataset index if found, otherwise ``None``.
        """
        for idx, (_, stem, key) in enumerate(self.tiles):
            if stem == image_stem and key == group_name:
                return idx
        return None

    def get_tile_metadata(self, idx: int) -> tuple[str, str]:
        """
        Return the (image_stem, group_name) identity tuple for a given index.

        Args:
            idx: Integer index in [0, len(dataset)).

        Returns:
            Tuple of (image_stem, group_name), e.g. ``("slide01", "tile_3_7")``.
        """
        _, image_stem, group_name = self.tiles[idx]
        return image_stem, group_name
