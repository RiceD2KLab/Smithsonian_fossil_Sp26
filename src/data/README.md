This directory provides core utilities for reading, parsing, and converting data from NDPI (Hamamatsu whole slide image) and NDPA (annotation) files, as well as nanometer-to-pixel conversions using NDPI metadata.

## Overview
- The **ndpi_reader.py** and **ndpa_reader.py** files provide structured, object-oriented access to slide and annotation data
- **ndpi_reader.py**: Defines the `NDPIData` class for loading NDPI files, extracting metadata (microns-per-pixel, objective power, offsets, image size), and mapping focal planes (magnification × z-offset) to TIFF pages. Provides methods to extract image tiles at specific magnifications and focal depths.
    - Example: `NDPIData(ndpi_path)` loads the file and exposes metadata and functionality to extract image tiles.
- **ndpa_reader.py**: Defines the `NDPAData` class for parsing NDPA XML annotation files. Extracts both large rectangular regions of interest (ROIs) and individual palynomorph annotations (typically circles), with bounding region classes for both. Provides structured access to annotation geometry and labels.
    - Example: `NDPAData(ndpa_path)` parses the file and exposes lists of ROIs and palynomorphs.
- **util.py**: Provides conversion utilities, including `bounds_to_pixels`, which converts annotation bounding regions (in nanometers) to pixel coordinates at a given magnification, using NDPI metadata.

## Example Usage

```python
from src.data.ndpi_reader import NDPIData
from src.data.ndpa_reader import NDPAData
from src.data.util import bounds_to_pixels

# Load NDPI metadata and focal planes
ndpi = NDPIData("slide.ndpi")

# Parse NDPA annotations
ndpa = NDPAData("slide.ndpa")

# Convert a bounding region to pixel coordinates at 20x
bbox_px = bounds_to_pixels(ndpa.palynomorphs[0].bounds, 20.0, ndpi.metadata)
```
