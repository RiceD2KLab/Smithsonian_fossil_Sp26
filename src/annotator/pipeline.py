"""
End-to-end NDPI annotation pipeline.

This module orchestrates tile extraction, tile compression, model inference,
cross-tile deduplication, and export to CSV, NDPA, and optional PNG crops.
"""

import csv
import math
import os
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from src.annotator.compression import get_compression_func
from src.annotator.config import AnnotatorConfig
from src.annotator.detectors import TileDetector, build_detector
from src.annotator.nms import deduplicate
from src.annotator.types import Detection, TileSpec
from src.data.ndpa_reader import NDPAData
from src.data.ndpa_writer import NDPAWriter
from src.data.ndpi_reader import NDPIData
from src.data.util import bounds_to_pixels, pixels_to_nm_bbox

import numpy as _np
from torch.utils.data import Dataset as _TorchDataset, DataLoader as _TorchDataLoader

# Per-process NDPI handle for DataLoader workers. Each worker process opens the
# slide once (not per band). Workers do CPU read+compress only; never touch CUDA.
_WORKER = {}

def _worker_state(ndpi_path, method, fsks, tks):
    st = _WORKER.get("state")
    if st is None:
        from src.data.ndpi_reader import NDPIData as _NDPIData
        from src.annotator.compression import get_compression_func as _gcf
        st = {
            "ndpi": _NDPIData(ndpi_path),
            "compress": _gcf(method=method, focus_stack_kernel_size=fsks, tenengrad_ksize=tks),
        }
        _WORKER["state"] = st
    return st


def _identity_collate(b):
    # one band per batch; return its list of (idx,x,y,w,h,tile_2d)
    return b[0]


class _BandDataset(_TorchDataset):
    """Indexed by band. __getitem__ reads one band, slices, compresses each tile.

    Returns a list of (tile_index_in_grid, x, y, w, h, tile_2d uint8). The
    tile_index lets the main process rebuild the exact TileSpec for keying and
    coordinate mapping. All work here is CPU-only (read + Tenengrad compress).
    """
    def __init__(self, ndpi_path, magnification, method, fsks, tks, bands, tile_index):
        self.ndpi_path = ndpi_path
        self.magnification = magnification
        self.method = method
        self.fsks = fsks
        self.tks = tks
        self.bands = bands              # list of (bx,by,bw,bh,[(idx,x,y,w,h),...])
        self.tile_index = tile_index    # unused at worker; kept for clarity

    def __len__(self):
        return len(self.bands)

    def __getitem__(self, i):
        st = _worker_state(self.ndpi_path, self.method, self.fsks, self.tks)
        ndpi = st["ndpi"]; compress = st["compress"]
        bx, by, bw, bh, tiles = self.bands[i]
        band_3d = ndpi.get_tile(x=bx, y=by, w=bw, h=bh, magnification=self.magnification)
        out = []
        for (idx, x, y, w, h) in tiles:
            off = x - bx
            tile_3d = band_3d[:, off:off + w, :, :]
            tile_2d = compress(tile_3d)
            out.append((idx, x, y, w, h, _np.ascontiguousarray(tile_2d)))
        return out


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
            tile_size=config.tile_size,
            rfdetr_variant=config.rfdetr_variant,
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
        Build the tile grid that covers the slide (or only the NDPA ROIs, if configured) at the target magnification.


        Returns: A list of `TileSpec` objects covering the target area in scan order.
        """
        image_w, image_h = self.ndpi.get_image_size_at_magnification(self.config.magnification)
        tile_size = self.config.tile_size
        stride =  max(1, int(round(tile_size * (1.0 - self.config.overlap))))

        if self.config.ndpa_path is None:
            xs = self._axis_positions(image_w, tile_size, stride)
            ys = self._axis_positions(image_h, tile_size, stride)
            return [
                TileSpec(x=x, y=y, w=tile_size, h=tile_size, ix=ix, iy=iy)
                for iy, y in enumerate(ys)
                for ix, x in enumerate(xs)
            ]

        ndpa = NDPAData(self.config.ndpa_path)
        tiles: list[TileSpec] = []
        iy_offset = 0
        for roi in ndpa.rois:
            roi_x, roi_y, roi_w, roi_h = bounds_to_pixels(roi.bounds, self.config.magnification, self.ndpi.metadata)
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_w = min(roi_w, image_w - roi_x)
            roi_h = min(roi_h, image_h - roi_y)
            if roi_w <= 0 or roi_h <= 0:
                continue
            xs = self._axis_positions(roi_w, tile_size, stride)
            ys = self._axis_positions(roi_h, tile_size, stride)
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    tiles.append(TileSpec(x=roi_x + x, y=roi_y + y, w=tile_size, h=tile_size,
                                          ix=ix, iy=iy_offset + iy))
            iy_offset += len(ys) + 1  # keeps per-ROI tiles contiguous in sort order so pixel-based NMS scan stops correctly
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
            details = f"source={self.config.model_name}; confidence={float(det.score):.6f}"
            writer.add_circle(
                label=self.config.annotation_class,
                lens=self.config.magnification,
                x_nm=det.x_nm,
                y_nm=det.y_nm,
                width_nm=det.width_nm,
                height_nm=det.height_nm,
                color=self.config.model_config.ndpa_color,
                details=details,
            )
        writer.save()

    def _write_crops(
        self,
        detections: list[Detection],
        output_dir: str
    ) -> None:
        """
        Write one PNG crop per detection.

        Args:
            detections: Final deduplicated detections to export.
            output_dir: Output directory for crops.
        """
        crop_dir = Path(output_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)

        for index, det in enumerate(detections, start=1):
            x0 = int(math.floor(det.x1_px))
            y0 = int(math.floor(det.y1_px))
            x1 = int(math.ceil(det.x2_px))
            y1 = int(math.ceil(det.y2_px))
            crop_w = max(1, x1 - x0)
            crop_h = max(1, y1 - y0)

            tile_3d = self.ndpi.get_tile(
                x=x0,
                y=y0,
                w=crop_w,
                h=crop_h,
                magnification=self.config.magnification,
            )
            crop_2d = self.compress(tile_3d)
            crop_path = crop_dir / f"{index:05d}.png"
            Image.fromarray(crop_2d).save(crop_path)

    def _group_into_bands(self, tiles, max_band_w):
        """Group consecutive tiles sharing (y, h) into bands no wider than max_band_w.

        A band is (band_x, band_y, band_w, band_h, [tiles]). Reading the band once
        and slicing decodes each underlying JPEG strip a single time instead of once
        per overlapping tile. Output is identical to per-tile reads.
        """
        bands = []
        i = 0
        n = len(tiles)
        while i < n:
            t0 = tiles[i]
            y, h = t0.y, t0.h
            band_x0 = t0.x
            group = [t0]
            j = i + 1
            while j < n:
                tj = tiles[j]
                if tj.y != y or tj.h != h:
                    break
                # stop if adding tj would exceed the width cap
                if (tj.x + tj.w) - band_x0 > max_band_w:
                    break
                group.append(tj)
                j += 1
            band_x1 = max(t.x + t.w for t in group)
            bands.append((band_x0, y, band_x1 - band_x0, h, group))
            i = j
        return bands

    def run_parallel(self) -> list[Detection]:
        """Parallel variant: band read+compress in worker processes, GPU in main.

        Identical detections to run(); only the producer is parallelized.
        Controlled by PALYNET_NUM_WORKERS (>=1) and PALYNET_BAND_W.
        """
        image_name = Path(self.config.ndpi_path).stem
        output_dir = os.path.abspath(self.config.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{image_name}.csv")
        ndpa_out = self.config.ndpa_path if self.config.ndpa_path else os.path.join(output_dir, f"{image_name}.ndpi.ndpa")

        _t0 = time.perf_counter()
        tiles = self._build_tile_grid()
        t_tiling = time.perf_counter() - _t0
        n_tiles = len(tiles)

        # index tiles so workers can return a key the main process maps back to TileSpec
        by_index = {i: t for i, t in enumerate(tiles)}
        index_of = {id(t): i for i, t in enumerate(tiles)}

        max_band_w = int(os.environ.get("PALYNET_BAND_W", "8192"))
        raw_bands = self._group_into_bands(tiles, max_band_w)
        # convert TileSpec bands into picklable plain tuples carrying the grid index
        bands = []
        for bx, by, bw, bh, group in raw_bands:
            gt = [(index_of[id(t)], t.x, t.y, t.w, t.h) for t in group]
            bands.append((bx, by, bw, bh, gt))

        n_workers = int(os.environ.get("PALYNET_NUM_WORKERS", "0"))
        ds = _BandDataset(
            self.config.ndpi_path, self.config.magnification,
            self.config.compression_method, self.config.focus_stack_kernel_size,
            self.config.tenengrad_ksize, bands, by_index,
        )
        # Use 'fork' so workers share the parent's torch pages copy-on-write;
        # 'spawn' re-imports torch per worker and blows past h_vmem. Fork after
        # CUDA init is safe here ONLY because workers do pure CPU read+compress
        # and never touch CUDA (the detector stays in the main process). If a
        # worker ever needs CUDA, this must change.
        import multiprocessing as _mp
        _ctx = _mp.get_context("fork") if n_workers > 0 else None
        loader = _TorchDataLoader(
            ds, batch_size=1, num_workers=n_workers,
            collate_fn=_identity_collate,
            prefetch_factor=(2 if n_workers > 0 else None),
            persistent_workers=False,
            multiprocessing_context=_ctx,
        )

        detections_by_tile: dict[TileSpec, list[Detection]] = {}
        t_compress = 0.0  # folded into worker time; reported as read for simplicity
        t_infer = 0.0
        t_read = 0.0

        with open(csv_path, "w", newline="") as checkpoint_file:
            checkpoint_writer = csv.DictWriter(checkpoint_file, fieldnames=CSV_FIELDNAMES)
            checkpoint_writer.writeheader()

            _tprod = time.perf_counter()
            for band_tiles in tqdm(loader, desc="Annotating tiles", unit="band"):
                t_read += time.perf_counter() - _tprod
                for (idx, x, y, w, h, tile_2d) in band_tiles:
                    tile = by_index[idx]
                    _t = time.perf_counter()
                    boxes, scores = self.detector.predict(tile_2d)
                    t_infer += time.perf_counter() - _t

                    if len(boxes) == 0:
                        continue
                    tile_candidates: list[Detection] = []
                    for box, score in zip(boxes, scores):
                        gx1 = float(x + box[0]); gy1 = float(y + box[1])
                        gx2 = float(x + box[2]); gy2 = float(y + box[3])
                        x_nm, y_nm, width_nm, height_nm = pixels_to_nm_bbox(
                            gx1, gy1, gx2, gy2,
                            self.config.magnification, self.ndpi.metadata,
                        )
                        tile_candidates.append(Detection(
                            x1_px=gx1, y1_px=gy1, x2_px=gx2, y2_px=gy2,
                            score=float(score), x_nm=x_nm, y_nm=y_nm,
                            width_nm=width_nm, height_nm=height_nm,
                        ))
                    detections_by_tile[tile] = tile_candidates
                    if tile_candidates:
                        checkpoint_writer.writerows(
                            [self._detection_to_csv_row(det) for det in tile_candidates]
                        )
                        checkpoint_file.flush()
                _tprod = time.perf_counter()

        n_detections = sum(len(v) for v in detections_by_tile.values())
        print(f"Total detections before NMS: {n_detections}")
        _t = time.perf_counter()
        final_detections = deduplicate(
            detections_by_tile=detections_by_tile,
            iou_threshold=self.config.nms_iou_threshold,
        )
        t_nms = time.perf_counter() - _t
        print(f"Total detections after NMS: {len(final_detections)}")

        per = lambda t: f"~{t / n_tiles:.3f}s/tile" if n_tiles else ""
        print("\n--- Pipeline Timing (parallel) ---")
        print(f"  Tiling:      {t_tiling:6.2f}s  ({n_tiles} tiles, {n_workers} workers)")
        print(f"  Produce(read+compress, wall): {t_read:6.2f}s  {per(t_read)}")
        print(f"  Inference:   {t_infer:6.2f}s  {per(t_infer)}")
        print(f"  NMS:         {t_nms:6.2f}s")
        print("-----------------------\n")

        self._write_csv(final_detections, csv_path)
        self._write_ndpa(final_detections, ndpa_out)
        if self.config.export_crops:
            self._write_crops(final_detections, os.path.join(output_dir, "crops"))
        return final_detections

    def run(self) -> list[Detection]:
        """
        Execute the full NDPI annotation pipeline.

        Returns:
            The final list of deduplicated detections.
        """

        # Dispatch to the parallel producer path when workers are requested.
        if int(os.environ.get("PALYNET_NUM_WORKERS", "0")) >= 1:
            return self.run_parallel()

        # Get NDPI image name and prepare output paths.
        image_name = Path(self.config.ndpi_path).stem
        output_dir = os.path.abspath(self.config.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{image_name}.csv")
        ndpa_out = self.config.ndpa_path if self.config.ndpa_path else os.path.join(output_dir, f"{image_name}.ndpi.ndpa")

        _t0 = time.perf_counter()
        tiles = self._build_tile_grid()
        t_tiling = time.perf_counter() - _t0
        n_tiles = len(tiles)

        detections_by_tile: dict[TileSpec, list[Detection]] = {}
        t_compress = 0.0
        t_infer = 0.0
        t_read = 0.0

        # Write checkpoint rows as tiles are processed; final CSV is rewritten after deduplication.
        with open(csv_path, "w", newline="") as checkpoint_file:
            checkpoint_writer = csv.DictWriter(checkpoint_file, fieldnames=CSV_FIELDNAMES)
            checkpoint_writer.writeheader()

            import os as _os
            max_band_w = int(_os.environ.get("PALYNET_BAND_W", "8192"))
            bands = self._group_into_bands(tiles, max_band_w)
            for band_x, band_y, band_w, band_h, band_tiles in tqdm(
                bands, desc="Annotating tiles", unit="band"
            ):
                _t = time.perf_counter()
                band_3d = self.ndpi.get_tile(
                    x=band_x, y=band_y, w=band_w, h=band_h,
                    magnification=self.config.magnification,
                )
                t_read += time.perf_counter() - _t

                for tile in band_tiles:
                    x, y, w, h = tile.x, tile.y, tile.w, tile.h
                    off = x - band_x
                    tile_3d = band_3d[:, off:off + w, :, :]

                    _t = time.perf_counter()
                    tile_2d = self.compress(tile_3d)
                    t_compress += time.perf_counter() - _t

                    _t = time.perf_counter()
                    boxes, scores = self.detector.predict(tile_2d)
                    t_infer += time.perf_counter() - _t

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

                    detections_by_tile[tile] = tile_candidates

                    if tile_candidates:
                        checkpoint_writer.writerows(
                            [self._detection_to_csv_row(det) for det in tile_candidates]
                        )
                        checkpoint_file.flush()

        n_detections = sum(len(tile_detections) for tile_detections in detections_by_tile.values())
        print(f"Total detections before NMS: {n_detections}")

        _t = time.perf_counter()
        final_detections = deduplicate(
            detections_by_tile=detections_by_tile,
            iou_threshold=self.config.nms_iou_threshold,
        )
        t_nms = time.perf_counter() - _t
        print(f"Total detections after NMS: {len(final_detections)}")

        t_total = t_tiling + t_read + t_compress + t_infer + t_nms
        per = lambda t: f"~{t / n_tiles:.3f}s/tile" if n_tiles else ""
        print("\n--- Pipeline Timing ---")
        print(f"  Tiling:      {t_tiling:6.2f}s  ({n_tiles} tiles)")
        print(f"  Read tile:   {t_read:6.2f}s  {per(t_read)}")
        print(f"  Compression (Focus Stack/Best Focal Plane): {t_compress:6.2f}s  {per(t_compress)}")
        print(f"  Inference:   {t_infer:6.2f}s  {per(t_infer)}")
        print(f"  NMS:         {t_nms:6.2f}s")
        print(f"  Total:       {t_total:6.2f}s")
        print("-----------------------\n")

        # Replace checkpoint CSV with final deduplicated results.
        self._write_csv(final_detections, csv_path)
        self._write_ndpa(final_detections, ndpa_out)
        if self.config.export_crops:
            self._write_crops(final_detections, os.path.join(output_dir, "crops"))
        return final_detections
