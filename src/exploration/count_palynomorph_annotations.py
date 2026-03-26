from src.data.ndpa_reader import NDPAData

"""
Parse NDPA files to count the number of palynomorph annotations per image.
"""

if __name__ == "__main__":
    with open("ndpa_names.txt", 'r') as file:
        total_palynomorph_annotations = 0
        for file_name in file:
            file_name = file_name.strip()
            # Example usage: parse an NDPA file and print the annotations
            prefix = "/path/to/ndpa/files/"
            ndpa_file = f"{prefix}{file_name}"
            anns = NDPAData(ndpa_file)
            rois_count = len(anns.rois)
            palynomorphs_count = len(anns.palynomorphs)
            total_palynomorph_annotations += palynomorphs_count
            print(f"{ndpa_file} - ROIS: {rois_count} | Palynomorphs: {palynomorphs_count}")
    print("Total palynomorph annotations:", total_palynomorph_annotations)