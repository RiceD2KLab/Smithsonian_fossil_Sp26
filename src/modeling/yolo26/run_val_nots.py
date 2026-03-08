"""
Patches ultralytics NMS to use TorchNMS instead of torchvision.ops.nms
to avoid "torchvision::nms not available for CUDA backend" on NOTS
"""
import os
import sys

if os.environ.get("ULTRALYTICS_FORCE_TORCH_NMS", "1") == "1":
    import ultralytics.utils.nms as _nms_mod

    _orig_nms = _nms_mod.non_max_suppression

    def _patched_nms(*args, **kwargs):
        _tv = sys.modules.pop("torchvision", None)
        try:
            return _orig_nms(*args, **kwargs)
        finally:
            if _tv is not None:
                sys.modules["torchvision"] = _tv

    _nms_mod.non_max_suppression = _patched_nms

from src.modeling.yolo26.val import main

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "-m":
        sys.argv = [sys.argv[0]] + sys.argv[3:]
    main()
