"""
YOLO26 training, when --h5_root and --splits_json are provided, uses H5YOLODataset (streams from
.h5 focus-stacked tiles). Otherwise uses standard YOLO dataset from data= path (yaml / folder).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model, torch_distributed_zero_first
from ultralytics.utils import LOGGER, colorstr

from src.modeling.yolo26.dataset import H5YOLODataset

GLOBAL_ARGS = None

def _data_dict_from_metadata(metadata_path: str) -> dict[str, Any]:
    """Build YOLO data dict (names, nc) from tiles metadata.json."""
    with open(metadata_path) as f:
        meta = json.load(f)
    label_map = meta.get("label_map", {})
    if not label_map:
        return {"names": {0: "class0"}, "nc": 1}
    # label_map is category name -> index (0-based)
    names = {int(idx): str(name) for name, idx in label_map.items()}
    return {"names": names, "nc": len(names)}


def _default_data_dict(nc: int = 2) -> dict[str, Any]:
    return {"names": {i: f"class{i}" for i in range(nc)}, "nc": nc}


class H5DetectionTrainer(DetectionTrainer):
    """DetectionTrainer that uses H5YOLODataset when h5_root and splits_json are set in args."""

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        global GLOBAL_ARGS
        h5_root = getattr(GLOBAL_ARGS, "h5_root", None)
        splits_json = getattr(GLOBAL_ARGS, "splits_json", None)
        if h5_root and splits_json:
            with torch_distributed_zero_first(rank):
                gs = max(int(unwrap_model(self.model).stride.max()), 32)
                dataset = H5YOLODataset(
                    h5_root=h5_root,
                    splits_json_path=splits_json,
                    split_name=mode,
                    data=self.data,
                    cache_labels=getattr(GLOBAL_ARGS, "cache_labels", True),
                    single_cls=getattr(GLOBAL_ARGS, "single_cls", False),
                    imgsz=self.args.imgsz,
                    batch_size=batch_size,
                    augment=(mode == "train"),
                    hyp=self.args,
                    rect=(mode == "val"),
                    stride=gs,
                    pad=0.0 if mode == "train" else 0.5,
                    prefix=colorstr(f"{mode}: "),
                    task="detect",
                    classes=getattr(self.args, "classes", None),
                    fraction=self.args.fraction if mode == "train" else 1.0,
                )
            shuffle = mode == "train"
            if getattr(dataset, "rect", False) and shuffle:
                try:
                    import numpy as np
                    if not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
                        LOGGER.warning(
                            "'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False"
                        )
                        shuffle = False
                except Exception:
                    pass
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=shuffle,
                rank=rank,
                drop_last=getattr(self.args, "compile", False) and mode == "train",
            )
        return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26 model.")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="Model variant (e.g. yolo26s.pt).")
    parser.add_argument("--h5_root", type=str, default=None, help="Directory of .h5 tile files (enables H5 dataset).")
    parser.add_argument("--splits_json", type=str, default=None, help="Path to train_val_test.json (required if --h5_root).")
    parser.add_argument("--metadata", type=str, default=None, help="Path to tiles metadata.json for names/nc (optional).")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (used only when not using H5).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024, help="Input size (match tile size or 640).")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="yolo26")
    parser.add_argument("--cache_labels", action="store_true", default=True, help="Cache H5 label list (default True).")
    parser.add_argument("--no_cache_labels", action="store_false", dest="cache_labels")
    parser.add_argument("--single_cls", action="store_true", help="Map all palynomorph types to one class (detection only).")
    args = parser.parse_args()

    global GLOBAL_ARGS
    GLOBAL_ARGS = args

    use_h5 = bool(args.h5_root and args.splits_json)
    if use_h5:
        if not os.path.isdir(args.h5_root):
            raise FileNotFoundError(f"h5_root not found: {args.h5_root}")
        if not os.path.isfile(args.splits_json):
            raise FileNotFoundError(f"splits_json not found: {args.splits_json}")
        if args.single_cls:
            data_dict = {"names": {0: "palynomorph"}, "nc": 1}
        elif args.metadata and os.path.isfile(args.metadata):
            data_dict = _data_dict_from_metadata(args.metadata)
        else:
            data_dict = _default_data_dict()

        data_dict["path"] = os.path.abspath(args.h5_root)
        data_dict["train"] = "."
        data_dict["val"] = "."

        import yaml
        yaml_path = os.path.join(args.h5_root, "h5_dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data_dict, f)

        overrides = {
            "data": yaml_path,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": args.project,
            "name": args.name,
        }
        if args.device is not None:
            overrides["device"] = args.device
        model = YOLO(args.model)
        model.train(**overrides, trainer=H5DetectionTrainer)
    else:
        if not args.data:
            raise ValueError("Provide either (--h5_root and --splits_json) or --data (yaml / folder path).")
        overrides = {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": args.project,
            "name": args.name,
        }
        if args.device is not None:
            overrides["device"] = args.device
        model = YOLO(args.model)
        model.train(data=args.data, **overrides)


if __name__ == "__main__":
    main()
