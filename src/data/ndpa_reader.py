"""
Load and parse NDPA annotation files into structured Python objects.

Parses the companion NDPA XML file for an NDPI slide, extracting both
large regions of interest (rectangles) and individual palynomorph
annotations (circles) with their classification labels and bounding regions.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple, List

# Define the types of bounding regions that can be present in the NDPA annotations
class BoundingRegion:
    """Base class for bounding regions (circle or rectangle) in an NDPA file."""
    
    def get_bounding_box (self) -> Tuple[int, int, int, int]:
        """Return the bounding box as (x, y, width, height) in nanometers."""
        raise NotImplementedError
    
@dataclass
class Circle (BoundingRegion):
    """Circle bounding region defined by center and radius, all in nanometers.
    
    Attributes:
        cx: Center x in nanometers.
        cy: Center y in nanometers.
        radius: Radius in nanometers.
    """

    cx: int
    cy: int
    radius: int

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Return the bounding box as (x, y, width, height) in nanometers."""
        return (self.cx - self.radius, self.cy - self.radius,
                self.radius * 2, self.radius * 2)


@dataclass
class Rectangle (BoundingRegion):
    """Rectangular bounding region defined by top-left corner, width, and height in nanometers.
    
    Attributes:
        x: Top-left x in nanometers.
        y: Top-left y in nanometers.
        width: Width in nanometers.
        height: Height in nanometers.
    """

    x: int        # top-left x
    y: int        # top-left y
    width: int
    height: int

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Return the bounding box as (x, y, width, height) in nanometers."""
        return (self.x, self.y, self.width, self.height)

# Define the types of annotations that can be present in the NDPA file
@dataclass
class RegionOfInterest:
    """Large annotation denoting a broad area of interest.
    
    Attributes:
        id: Unique identifier for the region of interest.
        bounds: Bounding region of the region of interest.
    """
    id: int
    bounds: BoundingRegion

@dataclass
class PalynomorphAnnotation:
    """Small annotation for an individual palynomorph with a classification label.
    
    Attributes:
        id: Unique identifier for the palynomorph annotation.
        label: Label category, e.g. "paly", "pol", "spo".
        bounds: Bounding region of the palynomorph annotation.
        details: Free-text details string, e.g. "source=yolo; confidence=0.912".
    """
    id: int
    label: str # label category, e.g. "paly", "pol", "spo"
    bounds: BoundingRegion 
    details: str = ""

# Define the data stored in the NDPA file
@dataclass
class NDPAData:
    """
    Structured data extracted from an NDPA XML file.

    Attributes:
        rois: Larger regions of interest (rectangles) containing palynomorph annotations.
        palynomorphs: Individual palynomorph annotations with classification labels.
    """

    rois: List[RegionOfInterest]
    palynomorphs: List[PalynomorphAnnotation]

    def __init__ (self, ndpa_path: str):
        """
        Parse an NDPA XML file into rois and palynomorphs lists.

        The NDPA file is a companion to the NDPI slide containing user-drawn
        annotations. Each <ndpviewstate> element describes one annotation with
        its type, position, and label. Rectangles (stored as "freehand" with 4
        corner points) are treated as ROIs; circles are treated as palynomorphs.

        Args:
            ndpa_path: Path to the .ndpa XML annotation file.

        Returns:
            An `NDPAData` instance.
        """
        tree = ET.parse(ndpa_path)
        root = tree.getroot()
        
        self.rois = []
        self.palynomorphs = []

        for vs in root.findall("ndpviewstate"):

            ann_id = int(vs.get("id", 0))
            ann_elem = vs.find("annotation")
            if ann_elem is None:
                continue
            ann_type = ann_elem.get("type", "")

            if ann_type == "circle":
                
                label = vs.findtext("title", "").strip()
                details = vs.findtext("details", "").strip()
                cx = int(ann_elem.findtext("x", "0"))
                cy = int(ann_elem.findtext("y", "0"))
                radius = int(ann_elem.findtext("radius", "0"))
                circle = Circle(cx=cx, cy=cy, radius=radius)
                self.palynomorphs.append(PalynomorphAnnotation(
                    id=ann_id, label=label, bounds=circle, details=details
                ))
                
            elif ann_type == "freehand":
                
                # NDP.view stores rectangles as "freehand" with 4 corner points
                points = []
                for pt in ann_elem.findall(".//point"):
                    px = int(pt.findtext("x", "0"))
                    py = int(pt.findtext("y", "0"))
                    points.append((px, py))
                    
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    rectangle = Rectangle(x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys))
                    self.rois.append(RegionOfInterest(
                        id=ann_id, bounds=rectangle
                    ))

if __name__ == "__main__":
    # Example usage: parse an NDPA file and print the annotations
    ndpa_file = "< file path >"
    anns = NDPAData(ndpa_file)
    for ann in anns.rois:
        print(f"Annotation {ann.id} (ROI): bbox={ann.bounds.get_bounding_box()}")
    for ann in anns.palynomorphs:
        print(f"Annotation {ann.id} ({ann.label}): bbox=({ann.bounds.get_bounding_box()})")