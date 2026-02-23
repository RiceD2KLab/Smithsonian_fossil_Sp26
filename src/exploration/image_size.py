import sys
import os
import gc
import openslide
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.data.ndpa_reader import NDPAData

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
Extract the width and height of all regions of interest from an NDPA file
"""
def extract_rois(path):
    try:
        annotations = NDPAData(path)
        bounding_box_dimensions = [r.bounds.get_bounding_box()[2:] for r in annotations.rois]
        return bounding_box_dimensions
    except Exception as e:
        print(f"Error parsing NDPA file {path}: {e}")
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

"""
Produce density plots for the width and height of all regions of interest given a directory containing NDPA files

Output: Format and write to disk a matplotlib density plot as a png

Example: plot_image_annotation_distributions("/rhf/allocations/dsci435/smithsonian_full_sp26/Images_with_annotations_for_CNN_training/")
"""
def plot_image_annotation_distributions(directory):
    roi_pixel_width = []
    roi_pixel_height = []
    roi_microns_width = []
    roi_microns_height = []

    # Gather sizes
    dir = os.scandir(directory)
    i = 0
    for entry in dir:
        if entry.is_dir():
            continue

        extension = entry.name.split(".")[-1]
        if extension != "ndpa":
            continue

        split_file_path = entry.path.split(".")
        ndpi_file_path = split_file_path[0] + "." + split_file_path[1]
        _,_,mpp_x,mpp_y = extract_metadata(ndpi_file_path)
        bounding_box_dimensions = extract_rois(entry.path)
        for tuple in bounding_box_dimensions:
            microns_w, microns_h = tuple[0] / 1000, tuple[1] / 1000
            pixels_w, pixels_h = microns_w / mpp_x, microns_h / mpp_y
            roi_pixel_width.append(pixels_w)
            roi_pixel_height.append(pixels_h)
            roi_microns_width.append(microns_w)
            roi_microns_height.append(microns_h)

        i += 1
        if i % 10 == 0:
            print("Gathered data from " + str(i) + " NDPA files")
            gc.collect()

    df = pd.DataFrame({
        'Pixels': np.concatenate([np.array(roi_pixel_width), np.array(roi_pixel_height)]),
        'Dimension': ['Width'] * len(roi_pixel_width) + ['Height'] * len(roi_pixel_height)
    })
    generate_density_plots(df, "Pixels", "ROI Pixel Dimensions Density Plot", "roi_pixel_dimensions_density_plot.png")

    df = pd.DataFrame({
        'Microns': np.concatenate([np.array(roi_microns_width), np.array(roi_microns_height)]),
        'Dimension': ['Width'] * len(roi_microns_width) + ['Height'] * len(roi_microns_height)
    })
    generate_density_plots(df, "Microns", "ROI Microns Dimensions Density Plot", "roi_microns_dimensions_density_plot.png")