"""
Extract small-object circle annotation diameters from NDP.view .ndpa files
and plot TWO histograms:
  1) Diameter in pixels
  2) Diameter in microns (µm)

Assumptions (confirmed by your sample):
- Small objects are stored as:
    <annotation type="circle" displayname="AnnotateCircle"> ... <radius>13700</radius> </annotation>
- coordformat is nanometers, so radius is in nanometers.
- All slides share the same scanning resolution (hard-coded MPP).

Outputs (created locally):
- circle_sizes_summary.csv
- small_annotation_diameter_2plots.png
"""

import glob
import xml.etree.ElementTree as ET

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# HARD-CODED SCANNER RESOLUTION
# -----------------------------
MPP_X = 0.22885913720105275  # µm / pixel
MPP_Y = 0.22886437497139195  # µm / pixel
NM_PER_PX_X = MPP_X * 1000.0
NM_PER_PX_Y = MPP_Y * 1000.0


# -----------------------------
# Filters (tune if needed)
# -----------------------------
MIN_DIAMETER_UM = 1      # ignore tiny junk
MAX_DIAMETER_UM = 250    # "small objects" cutoff (keeps plots clean)


def strip_ns(tag: str) -> str:
    """Remove XML namespace if present."""
    return tag.split("}")[-1]


def parse_circle_annotations(path: str):
    """
    Extract diameters from:
      <annotation type="circle" displayname="AnnotateCircle">
          <x>...</x>
          <y>...</y>
          <radius>...</radius>
      </annotation>

    Radius is in nanometers -> diameter_nm = 2 * radius_nm.
    """
    root = ET.parse(path).getroot()

    for ann in root.iter():
        if strip_ns(ann.tag) != "annotation":
            continue

        if ann.attrib.get("displayname") != "AnnotateCircle":
            continue

        # extra safety: your sample shows type="circle"
        if ann.attrib.get("type") != "circle":
            continue

        # IMPORTANT: only read <radius> that is a direct child of this annotation
        radius_nm = None
        for child in list(ann):
            if strip_ns(child.tag) == "radius":
                txt = (child.text or "").strip()
                try:
                    radius_nm = float(txt)
                except Exception:
                    radius_nm = None
                break

        if radius_nm is None:
            continue

        diameter_nm = 2.0 * radius_nm
        diameter_um = diameter_nm / 1000.0

        # Keep only "small" objects (no log scale)
        if not (MIN_DIAMETER_UM <= diameter_um <= MAX_DIAMETER_UM):
            continue

        # Pixels (use x/y mpp)
        diameter_px_x = diameter_nm / NM_PER_PX_X
        diameter_px_y = diameter_nm / NM_PER_PX_Y
        diameter_px = (diameter_px_x + diameter_px_y) / 2.0

        yield {
            "ndpa_file": path,
            "radius_nm": radius_nm,
            "diameter_um": diameter_um,
            "diameter_px": diameter_px,
        }


def main():
    files = sorted(glob.glob("*.ndpi.ndpa"))
    print("NDPA files found:", len(files))
    if not files:
        raise SystemExit("No NDPA files found in this folder.")

    rows = []
    for f in files:
        rows.extend(list(parse_circle_annotations(f)))

    if not rows:
        raise SystemExit(
            "No circle annotations extracted after filtering.\n"
            "Try increasing MAX_DIAMETER_UM or confirm your NDPA uses "
            'type="circle" displayname="AnnotateCircle" with <radius>.'
        )

    df = pd.DataFrame(rows)
    print("Total circles extracted:", len(df))
    print("Diameter (µm) min/median/max:",
          df["diameter_um"].min(),
          df["diameter_um"].median(),
          df["diameter_um"].max())

    # Save summary locally
    df.to_csv("circle_sizes_summary.csv", index=False)

    # -----------------------------
    # Plot: TWO sponsor-ready histograms
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["diameter_px"], bins=30, edgecolor="black")
    axes[0].set_title("Small Annotation Diameter (Pixels)")
    axes[0].set_xlabel("Diameter (pixels)")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["diameter_um"], bins=30, edgecolor="black")
    axes[1].set_title("Small Annotation Diameter (Microns)")
    axes[1].set_xlabel("Diameter (µm)")
    axes[1].set_ylabel("Count")

    # Clean sponsor-friendly style
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # fewer, cleaner ticks

    fig.suptitle(f"Distribution of Small Annotation Diameters (N={len(df)})",
                 fontsize=14)

    plt.tight_layout()
    plt.savefig("small_annotation_diameter_2plots.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()