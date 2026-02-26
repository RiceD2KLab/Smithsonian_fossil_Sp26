from pathlib import Path
import pandas as pd
from src.data.ndpa_reader import NDPAData

def find_all_ndpa_files(root_dir):
    """Recursively find all .ndpa files under root_dir."""
    return list(Path(root_dir).rglob("*.ndpa"))

def get_label_counts(ndpa_dir):
    from collections import Counter
    label_counter = Counter()
    
    ndpa_files = find_all_ndpa_files(ndpa_dir)
    print(f"Found {len(ndpa_files)} NDPA files.")
    
    for ndpa_path in ndpa_files:
        try:
            ndpa = NDPAData(str(ndpa_path))
            for ann in getattr(ndpa, 'palynomorphs', []):
                label = getattr(ann, 'label', None)
                if label:
                    label_counter[label] += 1
        except Exception as e:
            print(f"Error reading {ndpa_path}: {e}")
    
    print("\nLabel counts (sorted by count, descending):")
    for label, count in label_counter.most_common():
        print(f"{label}: {count}")

    return label_counter.keys()

def print_missing_labels(ndpa_labels, annotation_map):
    """
    Print labels present in ndpa_labels but missing from the annotation map.
    """
    # Unique CSV labels
    csv_labels_original = list(annotation_map['Specimen_name'].dropna().unique())
    csv_labels = set(label.lower() for label in csv_labels_original)

    # Loop through NDPA labels and print those not found in CSV labels (case-insensitive)
    print(f"\nNDPA labels not found in CSV (case-insensitive check):")
    for ndpa_label in ndpa_labels:
        if ndpa_label.lower() not in csv_labels:
            print(ndpa_label)
    
if __name__ == "__main__":

    NDPA_DIR = "/path/to/ndpa/files"
    ANNOTATION_MAP_PATH = "/path/to/annotation_map.csv"
    annotation_map = pd.read_csv(ANNOTATION_MAP_PATH)
    ndpa_labels = get_label_counts(NDPA_DIR)

    print_missing_labels(ndpa_labels, annotation_map)
