from pathlib import Path
from src.data.ndpa_reader import NDPAData

def find_all_ndpa_files(root_dir):
    """Recursively find all .ndpa files under root_dir."""
    return list(Path(root_dir).rglob("*.ndpa"))

def main(root_dir):
    from collections import Counter
    label_counter = Counter()
    
    ndpa_files = find_all_ndpa_files(root_dir)
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
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python label_counts.py <ndpa_root_dir>")
    else:
        main(sys.argv[1])
