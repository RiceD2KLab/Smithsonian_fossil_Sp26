"""
- YAML configuration generation for Ultralytics YOLO with H5-backed datasets
- Data dictionary builders from metadata files
- Common constants and configuration helpers
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SINGLE_CLASS_NAME: str = "palynomorph"


def build_data_dict_from_metadata(metadata_path: str) -> dict[str, Any]:
    """
    Build a YOLO-compatible data dictionary from a tiles metadata JSON file.

    Assumes the metadata file contains a 'label_map' key mapping class
    names to integer indices (0-based).

    Args:
        metadata_path: Path to the metadata JSON file.

    Returns:
        Dictionary with 'names' (dict mapping index -> name) and 'nc' (number of classes).
    """
    with open(metadata_path, "r", encoding="utf-8") as file_handle:
        metadata_contents: dict[str, Any] = json.load(file_handle)

    # Extract the label_map: {class_name: class_index}
    label_map: dict[str, int] = metadata_contents.get("label_map", {})

    # Handle empty label_map by returning a single default class
    if not label_map:
        return {"names": {0: "class0"}, "nc": 1}

    index_to_name: dict[int, str] = {
        int(class_index): str(class_name)
        for class_name, class_index in label_map.items()
    }

    return {"names": index_to_name, "nc": len(index_to_name)}


def build_default_data_dict(num_classes: int) -> dict[str, Any]:
    """
    Build a default YOLO-compatible data dictionary with generic class names.

    Used when no metadata file is provided.

    Args:
        num_classes: Number of classes to generate names for.

    Returns:
        Dictionary with 'names' (dict mapping index -> name) and 'nc' (number of classes).
    """
    generic_names: dict[int, str] = {
        class_index: f"class{class_index}"
        for class_index in range(num_classes)
    }

    return {"names": generic_names, "nc": num_classes}


def build_single_class_data_dict() -> dict[str, Any]:
    """
    Build a YOLO-compatible data dictionary for single-class detection mode.

    All annotations are treated as a single class

    Returns:
        Dictionary with single class at index 0.
    """
    return {"names": {0: DEFAULT_SINGLE_CLASS_NAME}, "nc": 1}


def prepare_h5_yaml(
    h5_root: str,
    metadata_path: str | None,
    single_cls: bool,
) -> str:
    """
    Generate a temporary YAML configuration file for Ultralytics YOLO.

    This YAML points Ultralytics YOLO to the H5-backed dataset directory and specifies
    class names/counts. The actual data loading is handled by the H5YOLODataset,
    so the train/val/test paths are set to "." (placeholder, BUT NEEDED for YOLODataset API compatibility).

    Args:
        h5_root: Directory containing .h5 tile files.
        metadata_path: Optional path to tiles metadata.json file containing class names.
        single_cls: If True, all classes are mapped to a single class.

    Returns:
        Path to the generated h5_dataset.yaml file.
    """
    # Determine data dictionary based on configuration
    if single_cls:
        data_dict: dict[str, Any] = build_single_class_data_dict()

    elif metadata_path is not None and os.path.isfile(metadata_path):
        data_dict = build_data_dict_from_metadata(metadata_path)

    else:
        data_dict = build_default_data_dict(num_classes=2)

    # These paths are placeholders (needed for compatibility with the YOLODataset API);
    # H5YOLODataset handles the actual data loading
    absolute_h5_root: str = os.path.abspath(h5_root)
    data_dict["path"] = absolute_h5_root
    data_dict["train"] = "."
    data_dict["val"] = "."
    data_dict["test"] = "."

    yaml_output_path: Path = Path(absolute_h5_root) / "h5_dataset.yaml"

    with open(yaml_output_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(data_dict, yaml_file, default_flow_style=False)

    return str(yaml_output_path)
