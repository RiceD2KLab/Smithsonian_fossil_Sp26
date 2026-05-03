"""
Non-maximum suppression helpers for the annotator pipeline.

This module implements the boundary-aware deduplication pass used after tile
inference. It merges detections that overlap across neighboring tiles while
keeping the highest-confidence detection in each connected duplicate cluster.
"""

from typing import Mapping

from src.annotator.types import Detection, TileSpec

def _has_intersection(a: Detection, b: Detection) -> bool:
    """
    Return whether two detections overlap in pixel space.

    Args:
        a: First detection in pixel `xyxy` coordinates.
        b: Second detection in pixel `xyxy` coordinates.

    Returns:
        `True` when the boxes intersect with positive area, otherwise `False`.
    """
    return not (
        a.x2_px <= b.x1_px
        or a.x1_px >= b.x2_px
        or a.y2_px <= b.y1_px
        or a.y1_px >= b.y2_px
    )

def _iou_pair(a: Detection, b: Detection) -> float:
    """
    Compute the intersection-over-union for two detections.

    Args:
        a: First detection in pixel `xyxy` coordinates.
        b: Second detection in pixel `xyxy` coordinates.

    Returns:
        The IoU score in the range `[0.0, 1.0]`.
    """
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


def _has_overlap(a: TileSpec, b: TileSpec) -> bool:
    """
    Return whether two tile rectangles overlap in image space.

    Args:
        a: First tile specification.
        b: Second tile specification.

    Returns:
        `True` when the tile rectangles intersect with positive area, otherwise
        `False`.
    """
    return not (
        a.x + a.w <= b.x
        or a.x >= b.x + b.w
        or a.y + a.h <= b.y
        or a.y >= b.y + b.h
    )


def _get_overlapping_tiles(
    tile_index: int,
    tiles: list[TileSpec],
) -> list[int]:
    """
    Return later tile indices whose rectangles overlap the given tile.

    Tiles are processed in row-major order. The scan stops once later tiles
    start below the current tile's bottom edge, which means they cannot
    overlap in image space.

    Args:
        tile_index: Index of the current tile in the row-major tile list.
        tiles: Row-major list of tile specifications.

    Returns:
        The indices of later tiles that overlap the tile at `tile_index`.
    """
    tile = tiles[tile_index]
    neighbors: list[int] = []

    for other_index in range(tile_index + 1, len(tiles)):
        other = tiles[other_index]
        if other.y >= tile.y + tile.h:
            break
        if _has_overlap(tile, other):
            neighbors.append(other_index)

    return neighbors


def deduplicate(
    detections_by_tile: Mapping[TileSpec, list[Detection]],
    iou_threshold: float,
) -> list[Detection]:
    """
    Deduplicate detections across overlapping tile boundaries.

    Args:
        detections_by_tile: Mapping from tile specifications to detections
            produced for those tiles. Tiles with no detections can be omitted.
        iou_threshold: IoU threshold above which detections are considered
            duplicates.

    Returns:
        The deduplicated detections.
    """
    if not detections_by_tile:
        return []

    # Process tiles in row-major order so neighbor search can stop early by Y.
    tiles = sorted(detections_by_tile.keys(), key=lambda t: (t.iy, t.ix))

    # Flatten all detections into one list and remember global detection ids per tile.
    all_detections: list[Detection] = []
    ids_by_tile: list[list[int]] = []

    for tile in tiles:
        dets = detections_by_tile.get(tile, [])
        ids: list[int] = []
        for det in dets:
            ids.append(len(all_detections))
            all_detections.append(det)
        ids_by_tile.append(ids)

    if not all_detections:
        return []

    # Union-find tracks connected duplicate groups across overlapping tiles.
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

    for tile_index, tile in enumerate(tiles):
        ids_a = ids_by_tile[tile_index] if tile_index < len(ids_by_tile) else []
        if not ids_a:
            continue

        # Compare only with later tiles that geometrically overlap this tile.
        for other_index in _get_overlapping_tiles(tile_index, tiles):
            ids_b = ids_by_tile[other_index] if other_index < len(ids_by_tile) else []
            if not ids_b:
                continue

            # Merge detection ids when boxes intersect and IoU is above threshold.
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
        
        # Keep the highest-confidence representative per duplicate component.
        if det.score > all_detections[best_id].score:
            best_by_component[root] = det_id

    # Emit one representative detection per connected duplicate group.
    final_detections = [all_detections[idx] for idx in best_by_component.values()]
    return final_detections
