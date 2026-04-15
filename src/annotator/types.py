"""Shared data structures for the annotator package."""

from dataclasses import dataclass

@dataclass(frozen=True)
class Detection:
    """Single detection in both pixel and physical-coordinate spaces."""

    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    score: float
    x_nm: float
    y_nm: float
    width_nm: float
    height_nm: float


@dataclass(frozen=True)
class TileSpec:
    """Grid tile coordinates with both pixel origin and grid index."""

    x: int
    y: int
    w: int
    h: int
    ix: int
    iy: int
