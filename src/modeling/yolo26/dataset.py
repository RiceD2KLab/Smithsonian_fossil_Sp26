"""
H5-backed YOLO dataset for Ultralytics YOLO26 training.

Streams the focus-stacked tiles from the .h5 files. Bound boxes are only kept if valid 
(area >= 16 px^2, aspect ratio). The labels for YOLO are built from the original H5 and
optionally cached to avoid re-scanning the H5 files on every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from src.exploration.h5_utils import list_tile_jobs
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOCAL_RANK

# Prefix for H5-backed samples (so load_image can branch)
H5_PREFIX = "h5::"


def _valid_box(x: int, y: int, w: int, h: int) -> bool:
    """A box is valid if it has area >= 16 and no extreme aspect ratio."""
    if w <= 0 or h <= 0 or w * h < 16:
        return False
    if h <= 0.25 * w or w <= 0.25 * h:
        return False
    return True


def _xywh_pixel_to_normalized(
    bboxes_xywh: np.ndarray,
    labels: np.ndarray,
    height: int,
    width: int,
    single_cls: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter and convert XYWH pixel to YOLO normalized (center x,y, w,h). Returns (cls (n,1), bboxes (n,4))."""
    cls_list, box_list = [], []
    for (x, y, w, h), lab in zip(bboxes_xywh.tolist(), labels.tolist()):
        xi, yi, wi, hi = int(x), int(y), int(w), int(h)
        if not _valid_box(xi, yi, wi, hi):
            continue
        cx = (xi + wi / 2.0) / width
        cy = (yi + hi / 2.0) / height
        nw = wi / width
        nh = hi / height
        box_list.append([cx, cy, nw, nh])
        cls_list.append(0 if single_cls else (int(lab) if lab >= 0 else 0))
    cls_arr = np.array(cls_list, dtype=np.float32).reshape(-1, 1) if cls_list else np.zeros((0, 1), dtype=np.float32)
    box_arr = np.array(box_list, dtype=np.float32) if box_list else np.zeros((0, 4), dtype=np.float32)
    return cls_arr, box_arr


def _cache_path(h5_root: str, split_name: str, stems_hash: str, single_cls: bool = False) -> Path:
    suffix = "_single" if single_cls else "_multi"
    return Path(h5_root) / f".yolo_h5_{split_name}_{stems_hash[:12]}{suffix}.cache"


def _build_labels_from_h5(
    im_files: list[str],
    data: dict[str, Any],
    single_cls: bool = False,
) -> list[dict[str, Any]]:
    """Build YOLO label dicts from H5 for each entry in im_files (each must be h5::path::group)."""
    labels_out = []
    for im_file in im_files:
        if not im_file.startswith(H5_PREFIX):
            raise ValueError(f"Expected H5-backed image, got {im_file}")
        _, h5_path, group_name = im_file.split("::", 2)
        with h5py.File(h5_path, "r") as f:
            if group_name not in f or "focus_stacked" not in f[group_name]:
                continue
            g = f[group_name]
            bboxes = np.array(g["bboxes"])
            labs = np.array(g["labels"])
            shape = g["focus_stacked"].shape
            h, w = int(shape[0]), int(shape[1])
        if bboxes.size == 0 and labs.size == 0:
            cls_arr = np.zeros((0, 1), dtype=np.float32)
            bbox_arr = np.zeros((0, 4), dtype=np.float32)
        else:
            cls_arr, bbox_arr = _xywh_pixel_to_normalized(bboxes, labs, h, w, single_cls=single_cls)
        labels_out.append({
            "im_file": im_file,
            "shape": (h, w),
            "cls": cls_arr,
            "bboxes": bbox_arr,
            "segments": [],
            "keypoints": None,
            "normalized": True,
            "bbox_format": "xywh",
        })
    return labels_out


class H5YOLODataset(YOLODataset):
    """
    YOLO detection dataset that loads images and labels from H5 tile files.

    Uses focus_stacked (H,W,C) and bboxes/labels per tile. Tiles are selected
    by split via train_val_test.json (image stems). A label-list cache is optional and
    keyed by the h5_root + split + stems hash.
    """

    def __init__(
        self,
        h5_root: str,
        splits_json_path: str,
        split_name: str,
        data: dict[str, Any],
        cache_labels: bool = True,
        img_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._h5_root = os.path.abspath(h5_root)
        self._splits_json_path = os.path.abspath(splits_json_path)
        self._split_name = split_name
        self._cache_labels = cache_labels
        if img_path is None:
            img_path = self._h5_root
        # Disable image caching; we load from H5 on demand
        kwargs.setdefault("cache", None)
        kwargs.pop("task", None)
        super().__init__(img_path=img_path, data=data, task="detect", **kwargs)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Return a list of string identifiers for each tile 
        (e.g. "h5::/path/to/slide.h5::tile_0_0") built from
         list_tile_jobs(h5_root) filtered by split (using train_val_test.json).
        """
        with open(self._splits_json_path) as f:
            splits = json.load(f)
        stem_set = set(splits[self._split_name])
        all_tiles = list_tile_jobs(self._h5_root)
        tiles = [(p, s, g) for p, s, g in all_tiles if s in stem_set]
        return [f"{H5_PREFIX}{p}::{g}" for p, s, g in tiles]

    def get_labels(self) -> list[dict[str, Any]]:
        """Build the list-of-dict structure from H5: for each tile, open the H5, read
         `bboxes` and `labels`, apply the bbox validity rule (area >= 16, aspect ratio not extreme),
          convert to normalized xywh.  Optionally cache this list to a .cache file 
          (keyed by `h5_root` + splits hash) to avoid re-opening all H5s on every run.
        """
        with open(self._splits_json_path) as f:
            splits = json.load(f)
        stems_hash = hashlib.sha256(
            json.dumps(sorted(splits[self._split_name]), sort_keys=True).encode()
        ).hexdigest()
        cache_path = _cache_path(self._h5_root, self._split_name, stems_hash, getattr(self, "single_cls", False))

        if self._cache_labels and cache_path.is_file() and LOCAL_RANK in (-1, 0):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("hash") == stems_hash and "labels" in cached:
                    return cached["labels"]
            except Exception:
                pass

        labels = _build_labels_from_h5(
            self.im_files, self.data, single_cls=getattr(self, "single_cls", False)
        )
        if not labels:
            raise RuntimeError(
                f"No valid tiles found for split {self._split_name} in {self._h5_root}. "
                "Verify that H5 groups have 'focus_stacked', 'bboxes', and 'labels'."
            )
        self.im_files = [lb["im_file"] for lb in labels]

        if self._cache_labels and LOCAL_RANK in (-1, 0):
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump({"hash": stems_hash, "labels": labels}, f)
            except Exception:
                pass
        return labels

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        im_file = self.im_files[i]
        if not im_file.startswith(H5_PREFIX):
            return super().load_image(i, rect_mode=rect_mode)

        _, h5_path, group_name = im_file.split("::", 2)

        with h5py.File(h5_path, "r") as f:
            img = np.array(f[group_name]["focus_stacked"])

        # Ultralytics expects BGR (OpenCV convention for RGB images)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[0], img.shape[1]

        if self.augment:
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                self.buffer.pop(0)

        return img, (h, w), (h, w)
