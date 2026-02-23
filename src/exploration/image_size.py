import sys
import os
import gc
import openslide
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.data import ndpi_reader

"""
Extract metadata from an NDPI slide
"""
def extract_metadata(path):
    try:
        with openslide.OpenSlide(path) as slide:
            props = slide.properties
            w, h = slide.dimensions
            mpp_x = float(props.get("openslide.mpp-x", 0))
            mpp_y = float(props.get("openslide.mpp-y", 0))
            return w, h, mpp_x, mpp_y
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

"""
Generate facet grid density plots for the given pandas dataframe
"""
def generate_density_plots(df, x_label, title, output_file):
    g = sns.FacetGrid(df, row="Dimension", hue="Dimension", aspect=4, height=3, palette=["red", "blue"])
    g.map(sns.kdeplot, x_label, fill=False, alpha=1, lw=1)
    g.figure.suptitle(title)
    g.figure.tight_layout()
    g.figure.subplots_adjust(top=0.9)

    plt.savefig(output_file, format="png")
    plt.show()

"""
Plot the image size distributions for both physical and pixel space for all NDPI slides in the given directory

Output: Format and write to disk a matplotlib density plot as a png

Example: plot_image_size_distributions("/rhf/allocations/dsci435/smithsonian_full_sp26/Images_with_annotations_for_CNN_training/")
"""
def plot_image_size_distributions(directory):
    pixel_width = []
    pixel_height = []
    micron_width = []
    micron_height = []

    # Gather sizes
    dir = os.scandir(directory)
    i = 0
    for entry in dir:
        if entry.is_dir():
            continue

        extension = entry.name.split(".")[-1]
        if extension != "ndpi":
            continue

        w,h,mpp_x,mpp_y = extract_metadata(entry.path)
        microns_width, microns_height = mpp_x * w, mpp_y * h
        pixel_width.append(w)
        pixel_height.append(h)
        micron_width.append(microns_width)
        micron_height.append(microns_height)

        i += 1
        if i % 10 == 0:
            print("Gathered data from " + str(i) + " NDPI files")
            gc.collect()

    # Generate density plots for pixel dimensions
    df = pd.DataFrame({
        'Pixels': np.concatenate([np.array(pixel_width), np.array(pixel_height)]),
        'Dimension': ['Width'] * len(pixel_width) + ['Height'] * len(pixel_height)
    })
    generate_density_plots(df, "Pixels", "Pixel Dimension Density Plots", "slide_pixel_dimensions_density_plot.png")

    df = pd.DataFrame({
        'Microns': np.concatenate([np.array(micron_width), np.array(micron_height)]),
        'Dimension': ['Width'] * len(micron_width) + ['Height'] * len(micron_height)
    })
    generate_density_plots(df, "Microns", "Micron Dimension Density Plots", "slide_micron_dimensions_density_plot.png")
