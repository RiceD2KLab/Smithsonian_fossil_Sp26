This directory contains scripts and utilities for exploring, analyzing, and visualizing the properties of NDPI/NDPA slide data and their derived tile datasets. These scripts provide tools for data understanding, quality control, and exploratory analysis. The preprocessing pipeline must be run before executing these scripts.

## File Overview

- **annotation_class_distribution.py**: Plots the distribution of annotation classes (e.g., pol, spo, paly, din, alg, fun).
- **annotation_count_histogram.py**: Plots the distribution of the number of annotated palynomorphs per image.
- **annotation_size_distribution.py**: Extracts and visualizes the size distribution of palynomorph annotations from NDPA files, plotting histograms of diameters in pixels and microns.
- **compute_annotation_density.py**: Calculates annotation density for a single NDPI/NDPA pair, using ROI and annotation information.
- **count_palynomorph_annotations.py**: Loads NDPA files and outputs the number of palynomorph annotations per file.
- **explore_focus.py**: Computes and visualizes focus metrics (Variance of Laplacian, Tenengrad) for slide tiles across focal planes, writing results to CSV and generating plots.
- **focus_metrics.py**: Provides functions for computing sharpness/focus metrics (Variance of Laplacian, Tenengrad) for microscopy images.
- **image_metadata.py**: Extracts and visualizes metadata from NDPI slides, particularly related to staining.
- **image_size.py**: Extracts and analyzes the width and height of regions of interest (ROIs) from NDPA files, generating density plots.
- **label_counts.py**: Counts and summarizes the frequency of unique annotation labels in NDPA files, and reports missing labels.

## Usage
- These scripts are intended for interactive analysis, not as a pipeline.
- Use them to generate plots and summary statistics for data understanding and quality control.
- See each script for specific usage instructions and input requirements.
