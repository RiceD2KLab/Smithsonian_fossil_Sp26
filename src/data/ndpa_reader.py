import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple, List

"""
This file provides funtionality to load and interact with an NDPA file.
In particular, it can load an NDPA file, and parse the annotations into 
a structured data class representing both the larger ROIs (rectangles 
demarcating annotated regions) and the individual annotations of 
palynomorphs, with classification labels and bounding regions.
"""

# Define the types of bounding regions that can be present in the NDPA annotations
class BoundingRegion:
    """Base class for bounding regions (circle or rectangle) in an NDPA file."""
    
    def get_bounding_box (self) -> Tuple[int, int, int, int]:
        """Return the bounding box as (x, y, width, height) in nanometers."""
        raise NotImplementedError
    
@dataclass
class Circle (BoundingRegion):
    cx: int
    cy: int
    radius: int

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        return (self.cx - self.radius, self.cy - self.radius,
                self.radius * 2, self.radius * 2)


@dataclass
class Rectangle (BoundingRegion):
    x: int        # top-left x
    y: int        # top-left y
    width: int
    height: int

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

# Define the types of annotations that can be present in the NDPA file
@dataclass
class RegionOfInterest:
    """Large annotation denoting a broad area of interest."""
    id: int
    bounds: BoundingRegion

@dataclass
class PalynomorphAnnotation:
    """Small annotation for an individual palynomorph with a classification label."""
    id: int
    label: str # label category, e.g. "paly", "pol", "spo"
    bounds: BoundingRegion 

# Define the data stored in the NDPA file
@dataclass
class NDPAData:
    """
    Structured data extracted from an NDPA XML file.
    
    rois: List of larger regions of interest (ROIs) containing palynomorph annotations.
    palynomorphs: List of annotations marking individual palynomorphs.
    """
    rois: List[RegionOfInterest]
    palynomorphs: List[PalynomorphAnnotation]


"""
NOTE: My current assumption is that rectangles are used to denote ROIs
and circles are used to denote individual palynomorphs.  This is based on
typical usage of NDP.view, but may not be universally true.  If we encounter
cases where this assumption breaks down, we may need to add additional logic
to distinguish between ROI and palynomorph annotations, such as by checking the
size of the bounding box or by looking for specific label patterns.
"""
def parse_ndpa(ndpa_path: str) -> NDPAData:
    """
    Parse an NDPA XML file and return an NDPAData object.

    The NDPA file is a companion to the NDPI slide, containing user-drawn
    annotations (circles around palynomorphs, rectangles around regions of
    interest).  Each <ndpviewstate> element describes one annotation with
    its type, position, and label.
    """
    tree = ET.parse(ndpa_path)
    root = tree.getroot()
    
    rois = []
    palynomorphs = []

    for vs in root.findall("ndpviewstate"):

        ann_id = int(vs.get("id", 0))
        ann_elem = vs.find("annotation")
        if ann_elem is None:
            continue
        ann_type = ann_elem.get("type", "")

        if ann_type == "circle":
            
            label = vs.findtext("title", "").strip()
            cx = int(ann_elem.findtext("x", "0"))
            cy = int(ann_elem.findtext("y", "0"))
            radius = int(ann_elem.findtext("radius", "0"))
            circle = Circle(cx=cx, cy=cy, radius=radius)
            palynomorphs.append(PalynomorphAnnotation(
                id=ann_id, label=label, bounds=circle
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
                rois.append(RegionOfInterest(
                    id=ann_id, bounds=rectangle
                ))

    return NDPAData(rois=rois, palynomorphs=palynomorphs)

if __name__ == "__main__":
    # Example usage: parse an NDPA file and print the annotations
    ndpa_file = "<file path>"
    anns = parse_ndpa(ndpa_file)
    for ann in anns.rois:
        print(f"Annotation {ann.id} (ROI): bbox={ann.bounds.get_bounding_box()}")
    for ann in anns.palynomorphs:
        print(f"Annotation {ann.id} ({ann.label}): bbox=({ann.bounds.get_bounding_box()})")