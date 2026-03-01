import argparse
import csv
import os
import sys

from src.exploration.compute_annotation_density import compute_density

"""
Batch-runs compute_density() over a folder of NDPI/NDPA file pairs and
writes results to a CSV.

A valid pair is any .ndpi file that has a corresponding .ndpi.ndpa file
in the same folder (or vice versa). Files missing their counterpart are
skipped and reported at the end.
"""

def find_pairs(folder: str) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Scan a folder for matched .ndpi / .ndpi.ndpa pairs.

    Returns:

        pairs   : List of (ndpi_path, ndpa_path) tuples for valid pairs.
        orphans : List of file paths that have no counterpart.
    """
    all_files = set(os.listdir(folder))
    ndpi_files = {f for f in all_files if f.endswith(".ndpi")}
    ndpa_files = {f for f in all_files if f.endswith(".ndpi.ndpa")}

    pairs = []
    orphans = []

    for ndpi_name in ndpi_files:
        expected_ndpa = ndpi_name + ".ndpa"
        if expected_ndpa in ndpa_files:
            pairs.append((
                os.path.join(folder, ndpi_name),
                os.path.join(folder, expected_ndpa),
            ))
        else:
            orphans.append(os.path.join(folder, ndpi_name))

    # Any NDPA files with no matching NDPI
    for ndpa_name in ndpa_files:
        expected_ndpi = ndpa_name[:-5]  # strip ".ndpa"
        if expected_ndpi not in ndpi_files:
            orphans.append(os.path.join(folder, ndpa_name))

    return pairs, orphans


def main():
    parser = argparse.ArgumentParser(
        description="Batch-compute palynomorph annotation density for a folder of NDPI/NDPA pairs."
    )
    parser.add_argument("--folder",    required=True,  help="Folder containing .ndpi and .ndpi.ndpa files.")
    parser.add_argument("--output",    default="annotation_density.csv", help="Output CSV file path (default: annotation_density.csv).")
    parser.add_argument("--tile-size", type=int, default=1024,           help="Tile edge length in pixels (default: 1024).")
    args = parser.parse_args()

    # Find pairs
    pairs, orphans = find_pairs(args.folder)

    if not pairs:
        print("No valid NDPI/NDPA pairs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} valid pair(s), {len(orphans)} orphan(s).\n")

    # Process pairs and write CSV
    fieldnames = ["file_name", "n_annotations", "n_rois", "n_tiles", "density"]
    errors = []

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ndpi_path, ndpa_path in pairs:
            file_name = os.path.basename(ndpi_path)
            print(f"  Processing {file_name} ... ", end="", flush=True)

            try:
                result = compute_density(
                    ndpa_path=ndpa_path,
                    ndpi_path=ndpi_path,
                    tile_size=args.tile_size,
                )
                writer.writerow({
                    "file_name":     file_name,
                    "n_annotations": result["n_annotations"],
                    "n_rois":        result["n_rois"],
                    "n_tiles":       result["n_tiles"],
                    "density":       f"{result['density']:.6f}",
                })
                print("OK")

            except Exception as exc:
                print(f"FAILED ({exc})")
                errors.append((file_name, str(exc)))

    # Print results summary
    print(f"\nResults written to: {args.output}")

    if orphans:
        print(f"\n{len(orphans)} orphaned file(s) (no matching counterpart):")
        for path in orphans:
            print(f"  {os.path.basename(path)}")

    if errors:
        print(f"\n{len(errors)} file(s) failed to process:")
        for name, reason in errors:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()