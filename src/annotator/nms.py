"""Non-maximum suppression helpers for the annotator pipeline."""

from dataclasses import dataclass
from typing import Mapping, Protocol

@dataclass(frozen=True)
class TileSpec:
    """Grid tile coordinates with both pixel origin and grid index."""

    x: int
    y: int
    w: int
    h: int
    ix: int
    iy: int

class DetectionLike(Protocol):
    """Protocol for detection objects that expose the fields needed for IoU/NMS."""

    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    score: float


def _has_intersection(a: DetectionLike, b: DetectionLike) -> bool:
    """Return `True` when two detections overlap in pixel space."""
    return not (
        a.x2_px <= b.x1_px
        or a.x1_px >= b.x2_px
        or a.y2_px <= b.y1_px
        or a.y1_px >= b.y2_px
    )


def _iou_pair(a: DetectionLike, b: DetectionLike) -> float:
    """Compute IoU for a pair of xyxy detections."""
    inter_x1 = max(a.x1_px, b.x1_px)
    inter_y1 = max(a.y1_px, b.y1_px)
    inter_x2 = min(a.x2_px, b.x2_px)
    inter_y2 = min(a.y2_px, b.y2_px)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, a.x2_px - a.x1_px) * max(0.0, a.y2_px - a.y1_px)
    area_b = max(0.0, b.x2_px - b.x1_px) * max(0.0, b.y2_px - b.y1_px)
    denom = area_a + area_b - inter + 1e-7
    return inter / denom


def _tile_rects_overlap(a: TileSpec, b: TileSpec) -> bool:
    """Return `True` when two tile rectangles overlap in image space."""
    return not (
        a.x + a.w <= b.x
        or a.x >= b.x + b.w
        or a.y + a.h <= b.y
        or a.y >= b.y + b.h
    )


def _iter_forward_overlap_neighbors(
    tile: TileSpec,
    tiles_by_key: Mapping[tuple[int, int], TileSpec],
    overlap_span: int,
) -> list[tuple[int, int]]:
    """Return lexicographically forward neighboring tiles that can overlap `tile`."""
    neighbors: list[tuple[int, int]] = []

    for ny in range(tile.iy, tile.iy + overlap_span + 1):
        x_start = tile.ix + 1 if ny == tile.iy else tile.ix - overlap_span
        x_end = tile.ix + overlap_span
        for nx in range(x_start, x_end + 1):
            key = (ny, nx)
            other = tiles_by_key.get(key)
            if other is None:
                continue
            if _tile_rects_overlap(tile, other):
                neighbors.append(key)

    return neighbors


def deduplicate_tile_boundaries(
    tiles_by_key: Mapping[tuple[int, int], TileSpec],
    detections_by_tile: Mapping[tuple[int, int], list[DetectionLike]],
    iou_threshold: float,
    overlap_span: int,
) -> list[DetectionLike]:
    """Deduplicate only across overlapping tile pairs in one end-of-run pass."""
    all_detections: list[DetectionLike] = []
    ids_by_tile: dict[tuple[int, int], list[int]] = {}

    for key in sorted(tiles_by_key):
        dets = detections_by_tile.get(key, [])
        ids: list[int] = []
        for det in dets:
            ids.append(len(all_detections))
            all_detections.append(det)
        ids_by_tile[key] = ids

    if not all_detections:
        return []

    parent = list(range(len(all_detections)))
    rank = [0] * len(all_detections)

    def find(x: int) -> int:
        """Return the union-find root for detection index `x`."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        """Merge the union-find components containing `a` and `b`."""
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for key in sorted(tiles_by_key):
        tile = tiles_by_key[key]
        ids_a = ids_by_tile.get(key, [])
        if not ids_a:
            continue

        for nkey in _iter_forward_overlap_neighbors(tile, tiles_by_key, overlap_span):
            ids_b = ids_by_tile.get(nkey, [])
            if not ids_b:
                continue

            for ida in ids_a:
                det_a = all_detections[ida]
                for idb in ids_b:
                    det_b = all_detections[idb]
                    if not _has_intersection(det_a, det_b):
                        continue
                    if _iou_pair(det_a, det_b) > iou_threshold:
                        union(ida, idb)

    best_by_component: dict[int, int] = {}
    for det_id, det in enumerate(all_detections):
        root = find(det_id)
        best_id = best_by_component.get(root)
        if best_id is None:
            best_by_component[root] = det_id
            continue
        if det.score > all_detections[best_id].score:
            best_by_component[root] = det_id

    final_detections = [all_detections[idx] for idx in best_by_component.values()]
    final_detections.sort(key=lambda d: d.score, reverse=True)
    return final_detections
