"""Quick script-level checks for deduplicate in src.annotator.nms."""

from src.annotator.nms import deduplicate
from src.annotator.types import Detection, TileSpec


def _tile(x: int, y: int, w: int, h: int, ix: int, iy: int) -> TileSpec:
    return TileSpec(x=x, y=y, w=w, h=h, ix=ix, iy=iy)


def _det(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float,
) -> Detection:
    # Physical-space fields are not used by deduplicate, so mirror pixel values.
    return Detection(
        x1_px=x1,
        y1_px=y1,
        x2_px=x2,
        y2_px=y2,
        score=score,
        x_nm=x1,
        y_nm=y1,
        width_nm=x2 - x1,
        height_nm=y2 - y1,
    )


def test_deduplicate_empty() -> None:
    assert deduplicate({}, iou_threshold=0.5) == []


def test_deduplicate_merges_overlap_and_keeps_best_score() -> None:
    tile_a = _tile(0, 0, 100, 100, ix=0, iy=0)
    tile_b = _tile(80, 0, 100, 100, ix=1, iy=0)  # overlaps tile_a
    tile_c = _tile(0, 120, 100, 100, ix=0, iy=1)  # does not overlap tile_a

    # High-IoU pair across overlapping tiles: should merge and keep score=0.90.
    seam_low_score = _det(70, 30, 95, 60, score=0.60)
    seam_high_score = _det(72, 32, 97, 62, score=0.90)

    # Non-duplicates that should remain.
    unique_a = _det(10, 10, 20, 20, score=0.70)
    unique_c = _det(10, 130, 20, 140, score=0.80)

    deduped = deduplicate(
        {
            tile_a: [seam_low_score, unique_a],
            tile_b: [seam_high_score],
            tile_c: [unique_c],
        },
        iou_threshold=0.5,
    )

    scores = sorted(round(d.score, 2) for d in deduped)
    assert len(deduped) == 3
    assert scores == [0.7, 0.8, 0.9]


def main() -> None:
    test_deduplicate_empty()
    test_deduplicate_merges_overlap_and_keeps_best_score()
    print("All nms.deduplicate checks passed.")


if __name__ == "__main__":
    main()