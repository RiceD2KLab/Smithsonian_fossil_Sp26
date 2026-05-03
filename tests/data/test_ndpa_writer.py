import argparse
from pathlib import Path

from src.data.ndpa_reader import NDPAData
from src.data.ndpa_writer import NDPAWriter

def copy_palynomorphs(input_ndpa: str, output_root: str = "output") -> tuple[str, int]:
    """Copy palynomorph annotations from source NDPA to output NDPA."""
    input_path = Path(input_ndpa)
    if not input_path.exists():
        raise FileNotFoundError(f"Input NDPA file not found: {input_path}")
    output_path = Path(output_root) / input_path.name

    ndpa = NDPAData(str(input_path))
    writer = NDPAWriter(str(output_path))

    for ann in ndpa.palynomorphs:
        x_nm, y_nm, w_nm, h_nm = ann.bounds.get_bounding_box()
        writer.add_bounding_box(
            label=ann.label,
            lens=20.0,
            x_nm=x_nm,
            y_nm=y_nm,
            width_nm=w_nm,
            height_nm=h_nm,
            color="#0000ff",
            details="source=ground truth",
        )

    writer.save()
    return str(output_path), len(ndpa.palynomorphs)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy palynomorph annotations from an NDPA file into a new NDPA using NDPAWriter."
    )
    parser.add_argument("--input_ndpa", required=True, help="Path to source NDPA file.")
    parser.add_argument(
        "--output_root",
        default="output",
        help="Output directory. Output filename matches input NDPA name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path, copied = copy_palynomorphs(args.input_ndpa, args.output_root)
    print(f"Copied {copied} palynomorph annotations to {out_path}")


if __name__ == "__main__":
    main()
