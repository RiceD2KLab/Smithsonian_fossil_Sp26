"""
End-to-end NDPI annotation pipeline.

This module orchestrates tile extraction, tile compression, model inference,
cross-tile deduplication, and export to CSV and NDPA outputs.
"""

import csv
import os
from pathlib import Path
from tqdm import tqdm

from src.annotator.compression import get_compression_func
from src.annotator.config import AnnotatorConfig
from src.annotator.detectors import TileDetector, build_detector
from src.annotator.nms import deduplicate
from src.annotator.types import Detection, TileSpec
from src.data.ndpa_writer import NDPAWriter
from src.data.ndpi_reader import NDPIData
from src.data.util import pixels_to_nm_bbox

CSV_FIELDNAMES = [
    "class",
    "confidence",
    "x_nm",
    "y_nm",
    "width_nm",
    "height_nm",
    "x1_px",
    "y1_px",
    "x2_px",
    "y2_px",
]

class NDPIAnnotator:
    """Run end-to-end annotation for one NDPI slide.

    Args:
        config: Validated annotator configuration describing the input slide,
            model, tile settings, and output options.

    Returns:
        An `NDPIAnnotator` instance configured to annotate one slide.
    """

    def __init__(
        self,
        config: AnnotatorConfig,
    ) -> None:
        """
        Initialize the pipeline with configuration, metadata, and detector state.

        Args:
            config: Annotator configuration for the run.

        Throws error if configuration is invalid.
        """
        config.validate()
        self.config = config
        self.ndpi = NDPIData(config.ndpi_path)
        self.compress = get_compression_func(
            method=config.compression_method,
            focus_stack_kernel_size=config.focus_stack_kernel_size,
            tenengrad_ksize=config.tenengrad_ksize,
        )
        self.detector: TileDetector = build_detector(
            model_name=config.model_name,
            checkpoint_path=config.checkpoint_path,
            confidence_threshold=config.confidence_threshold,
            tile_size=config.model_config.tile_size,
        )

    @staticmethod
    def _axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
        """
        Return tile origins that cover one image axis.

        Args:
            length: Axis length in pixels.
            tile_size: Tile edge length in pixels.
            stride: Step size between adjacent tile origins in pixels.

        Returns:
            A list of tile origin coordinates that fully cover the axis.
        """
        if length <= tile_size:
            return [0]

        positions = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _build_tile_grid(self) -> list[TileSpec]:
        """
        Build the tile grid that covers the slide at the target magnification.
        
        Returns:
            A list of `TileSpec` objects covering the slide in scan order.
        """
        image_w, image_h = self.ndpi.get_image_size_at_magnification(self.config.magnification)
        tile_size = self.config.tile_size
        stride =  max(1, int(round(tile_size * (1.0 - self.config.overlap))))

        xs = self._axis_positions(image_w, tile_size, stride)
        ys = self._axis_positions(image_h, tile_size, stride)
        tiles: list[TileSpec] = []
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                tiles.append(TileSpec(x=x, y=y, w=tile_size, h=tile_size, ix=ix, iy=iy))
        return tiles
    
    def _detection_to_csv_row(self, det: Detection) -> dict[str, float | str]:
        """
        Convert one detection into a CSV row dictionary.

        Args:
            det: Detection object produced by the pipeline.

        Returns:
            A dictionary matching the CSV export schema.
        """
        return {
            "class": self.config.annotation_class,
            "confidence": float(det.score),
            "x_nm": float(det.x_nm),
            "y_nm": float(det.y_nm),
            "width_nm": float(det.width_nm),
            "height_nm": float(det.height_nm),
            "x1_px": float(det.x1_px),
            "y1_px": float(det.y1_px),
            "x2_px": float(det.x2_px),
            "y2_px": float(det.y2_px),
        }

    def _write_csv(self, detections: list[Detection], output_csv_path: str) -> None:
        """
        Write detections to a CSV file..

        Args:
            detections: List of detections to export.
            output_csv_path: Destination path for the CSV file.
        """
        with open(output_csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for det in detections:
                writer.writerow(
                    self._detection_to_csv_row(det)
                )

    def _write_ndpa(self, detections: list[Detection], output_ndpa_path: str) -> None:
        """
        Write detections to an NDPA file using circle annotations.

        Args:
            detections: List of detections to export.
            output_ndpa_path: Destination path for the NDPA file.
        """
        writer = NDPAWriter(output_path=output_ndpa_path)
        for det in detections:
            details = f"source={self.model_key}; confidence={float(det.score):.6f}"
            writer.add_circle(
                label=self.config.annotation_class,
                lens=self.config.magnification,
                x_nm=det.x_nm,
                y_nm=det.y_nm,
                width_nm=det.width_nm,
                height_nm=det.height_nm,
                color=self.model_config.ndpa_color,
                details=details,
            )
        writer.save()

    def run(self) -> list[Detection]:
        """
        Execute the full NDPI annotation pipeline.

        Returns:
            The final list of deduplicated detections.
        """

        # Get NDPI image name and prepare output paths.
        image_name = Path(self.config.ndpi_path).name.strip(".ndpi")
        output_dir = os.path.abspath(self.config.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{image_name}.csv")
        ndpa_path = os.path.join(output_dir, f"{image_name}.ndpi.ndpa")

        tiles = self._build_tile_grid()
        detections_by_tile: list[list[Detection]] = [[] for _ in tiles]

        # Write checkpoint rows as tiles are processed; final CSV is rewritten after deduplication.
        with open(csv_path, "w", newline="") as checkpoint_file:
            checkpoint_writer = csv.DictWriter(checkpoint_file, fieldnames=CSV_FIELDNAMES)
            checkpoint_writer.writeheader()

            for tile_index, tile in enumerate(tqdm(tiles, desc="Annotating tiles", unit="tile")):
                x, y, w, h = tile.x, tile.y, tile.w, tile.h
                tile_3d = self.ndpi.get_tile(x=x, y=y, w=w, h=h, magnification=self.config.magnification)
                tile_2d = self.compress(tile_3d)
                boxes, scores = self.detector.predict(tile_2d)

                if len(boxes) == 0:
                    continue

                tile_candidates: list[Detection] = []
                for box, score in zip(boxes, scores):
                    gx1 = float(x + box[0])
                    gy1 = float(y + box[1])
                    gx2 = float(x + box[2])
                    gy2 = float(y + box[3])

                    x_nm, y_nm, width_nm, height_nm = pixels_to_nm_bbox(
                        gx1,
                        gy1,
                        gx2,
                        gy2,
                        self.config.magnification,
                        self.ndpi.metadata,
                    )
                    tile_candidates.append(
                        Detection(
                            x1_px=gx1,
                            y1_px=gy1,
                            x2_px=gx2,
                            y2_px=gy2,
                            score=float(score),
                            x_nm=x_nm,
                            y_nm=y_nm,
                            width_nm=width_nm,
                            height_nm=height_nm,
                        )
                    )

                detections_by_tile[tile_index] = tile_candidates

                if tile_candidates:
                    checkpoint_writer.writerows(
                        [self._detection_to_csv_row(det) for det in tile_candidates]
                    )
                    checkpoint_file.flush()

        final_detections = deduplicate(
            tiles=tiles,
            detections_by_tile=detections_by_tile,
            iou_threshold=self.config.nms_iou_threshold,
        )

        # Replace checkpoint CSV with final deduplicated results.
        self._write_csv(final_detections, csv_path)
        self._write_ndpa(final_detections, ndpa_path)
        return final_detections
