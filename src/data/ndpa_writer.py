import os
import xml.etree.ElementTree as ET


class NDPAWriter:
    """Write NDPA annotations in XML format with rectangle freehand or circle entries."""

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self._tree, self._root = self._load_or_create(output_path)
        self._next_id = self._compute_next_id()
        if not os.path.exists(output_path):
            self.save()

    @staticmethod
    def _load_or_create(path: str) -> tuple[ET.ElementTree, ET.Element]:
        """
        Load an existing NDPA XML file or create a new one if it doesn't exist.
        """
        # Load existing NDPA XML if available
        if os.path.exists(path):
            tree = ET.parse(path)
            root = tree.getroot()
            if root.tag != "annotations":
                raise ValueError(f"Invalid NDPA root '{root.tag}'. Expected 'annotations'.")
        
        # Otherwise, create a new XML tree with the correct root
        else:
            root = ET.Element("annotations")
            tree = ET.ElementTree(root)

        # Return both the tree and root
        return tree, root

    def _compute_next_id(self) -> int:
        """
        Compute the next available annotation ID by scanning existing <ndpviewstate> entries.
        """
        max_id = 0
        for state in self._root.findall("ndpviewstate"):
            raw_id = state.get("id", "0")
            max_id = max(max_id, int(raw_id))
        return max_id + 1

    def add_bounding_box(
        self,
        label: str,
        lens: float,
        x_nm: float,
        y_nm: float,
        width_nm: float,
        height_nm: float,
        color: str = "#000000",
        details: str = "",
    ) -> int:
        """
        Add one rectangular annotation in NDPA format.

        Args:
            label: Class label used as <title>.
            lens: Magnification written to <lens>.
            x_nm: Top-left x in nanometers.
            y_nm: Top-left y in nanometers.
            width_nm: Bounding box width in nanometers.
            height_nm: Bounding box height in nanometers.
            color: Hex color for the annotation stroke.
            details: Free-text note stored in <details> (e.g. source metadata).

        Returns:
            The integer ID assigned to the new annotation.
        """
        x1 = int(round(x_nm))
        y1 = int(round(y_nm))
        x2 = int(round(x_nm + width_nm))
        y2 = int(round(y_nm + height_nm))

        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))

        state = ET.SubElement(self._root, "ndpviewstate", {"id": str(self._next_id)})
        self._next_id += 1

        ET.SubElement(state, "title").text = str(label)
        details_elem = ET.SubElement(state, "details")
        if details:
            details_elem.text = str(details)
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

    def add_circle(
        self,
        label: str,
        lens: float,
        x_nm: float,
        y_nm: float,
        width_nm: float,
        height_nm: float,
        color: str = "#000000",
        details: str = "",
    ) -> int:
        """
        Add one circular annotation in NDPA format.

        The circle is inscribed in the input bounding box, i.e. centered at the
        box center with radius = min(width, height) / 2.
        """
        x1 = int(round(x_nm))
        y1 = int(round(y_nm))
        x2 = int(round(x_nm + width_nm))
        y2 = int(round(y_nm + height_nm))

        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        radius = max(1, int(round(min(abs(width_nm), abs(height_nm)) / 2.0)))

        state = ET.SubElement(self._root, "ndpviewstate", {"id": str(self._next_id)})
        self._next_id += 1

        ET.SubElement(state, "title").text = str(label)
        details_elem = ET.SubElement(state, "details")
        if details:
            details_elem.text = str(details)
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
                "type": "circle",
                "displayname": "AnnotateCircle",
                "color": color,
            },
        )
        ET.SubElement(annotation, "x").text = str(cx)
        ET.SubElement(annotation, "y").text = str(cy)
        ET.SubElement(annotation, "radius").text = str(radius)
        ET.SubElement(annotation, "measuretype").text = "0"
        return self._next_id - 1

    def save(self) -> None:
        """Write NDPA XML to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        ET.indent(self._tree, space="\t")
        self._tree.write(self.output_path, encoding="utf-8", xml_declaration=True)
