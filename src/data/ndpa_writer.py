from __future__ import annotations

import os
import xml.etree.ElementTree as ET


class NDPAWriter:
    """Write NDPA annotations in XML format with rectangle freehand entries."""

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self._root = ET.Element("annotations")
        self._next_id = 1
        self.save()

    @staticmethod
    def _to_int(value: float | int) -> int:
        return int(round(float(value)))

    def add_bounding_box(
        self,
        label: str,
        lens: float,
        x_nm: float,
        y_nm: float,
        width_nm: float,
        height_nm: float,
        color: str = "#000000",
    ) -> int:
        """Add one rectangular annotation in NDPA format.

        Args:
            label: Class label used as <title>.
            lens: Magnification written to <lens>.
            x_nm: Top-left x in nanometers.
            y_nm: Top-left y in nanometers.
            width_nm: Bounding box width in nanometers.
            height_nm: Bounding box height in nanometers.
            color: Hex color for the annotation stroke.
        """
        x1 = self._to_int(x_nm)
        y1 = self._to_int(y_nm)
        x2 = self._to_int(x_nm + width_nm)
        y2 = self._to_int(y_nm + height_nm)

        cx = self._to_int((x1 + x2) / 2.0)
        cy = self._to_int((y1 + y2) / 2.0)

        state = ET.SubElement(self._root, "ndpviewstate", {"id": str(self._next_id)})
        self._next_id += 1

        ET.SubElement(state, "title").text = str(label)
        ET.SubElement(state, "details")
        ET.SubElement(state, "coordformat").text = "nanometers"
        ET.SubElement(state, "lens").text = f"{float(lens):.6f}"
        ET.SubElement(state, "x").text = str(cx)
        ET.SubElement(state, "y").text = str(cy)
        ET.SubElement(state, "z").text = "0"
        ET.SubElement(state, "showtitle").text = "0"
        ET.SubElement(state, "showhistogram").text = "0"
        ET.SubElement(state, "showlineprofile").text = "0"

        annotation = ET.SubElement(
            state,
            "annotation",
            {
                "type": "freehand",
                "displayname": "AnnotateRectangle",
                "color": color,
            },
        )
        ET.SubElement(annotation, "measuretype").text = "2"
        ET.SubElement(annotation, "closed").text = "1"

        pointlist = ET.SubElement(annotation, "pointlist")
        corners = ((x1, y1), (x1, y2), (x2, y2), (x2, y1))
        for px, py in corners:
            point = ET.SubElement(pointlist, "point")
            ET.SubElement(point, "x").text = str(px)
            ET.SubElement(point, "y").text = str(py)

        ET.SubElement(annotation, "specialtype").text = "rectangle"
        return self._next_id - 1

    def save(self, output_path: str | None = None) -> None:
        """Write NDPA XML to disk, creating parent directories as needed."""
        target = output_path if output_path is not None else self.output_path
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        tree = ET.ElementTree(self._root)
        ET.indent(tree, space="\t")
        tree.write(target, encoding="utf-8", xml_declaration=True)
