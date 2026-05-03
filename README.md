# Automated Detection of Palynomorphs from Digital Microscopy Slides

## Table of Contents

- Introduction
- Repo Structure
- Installation
- Data
- Data Preprocessing
- Exploratory Data Analysis
- Modeling & Evaluation
- Usage
  
## Introduction

### Team Members  

**Students:**    

- Abbas Shaikh
- Teon Golden
- Aditya Viswanathan
- Praise Mayor
- Patrick Ainlay-Vazquez
- Eric Zhang

**Faculty Mentor:**  

- Dr. Arko Barman

**PhD Mentor:** 

- Tony Yu

**Sponsors:**  

- Dr. Ingrid Romero
- Dr. Scott Wing

### Project Description

This repository contains code for the implementation of various objection detection models for the task of detecting palynomorphs (microfossils including pollen, spores, and algae) from digital microscopy data. It additionally provides an efficient and scalable pipeline for handling large-scale, multi-focal microscope images. In particular, the repository provides the functionality to tile the source dataset into machine learning ready-formats, perform various tasks like focus stacking to prepare image data, and train several state-of-the-art object detection architectures including YOLO26 and RF-DETR. More information on the various components of the repository can be found below.

## Repo Structure

- `scripts/` - Scripts for running experiments and launching SLURM jobs.
- `src/` — Main source code for the project, organized by functionality:
  - `src/annotator/` — An end-to-end, annotation pipeline.
  - `src/data/` — Utilities for reading, parsing, and converting raw microscopy data.
  - `src/exploration/` — Tools for exploratory data analysis and visualization.
  - `src/models/` — Code for model training of evaluated architectures, including YOLO26 and RF-DETR.
  - `src/plots/` — Scripts for generating result plots for model testing.
  - `src/preprocessing/` - Scripts for converting raw data into machine learning-ready formats.
- `tests/` — Unit tests for various code.
- `requirements.txt` — List of Python dependencies required to run the project.

## Installation

### Prerequisites
- Python >= 3.13

### Environment Setup

**Clone the repository:**
   
```bash
git clone --recurse-submodules https://github.com/RiceD2KLab/Smithsonian_fossil_Sp26.git
cd Smithsonian_fossil_Sp26
```

**Install dependencies:**
   
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Data

### Original Data Format
The original dataset consists of multi-focal microscopy images. Each sample is stored as an NDPI image file, which contains images of the sample at various magnifications and focal planes. Associated annotations are provided in separate, corresponding NDPA files, denoting the location of palynomorphs within the NDPI files. Our data preprocessing pipeline expects the NDPI and NDPA files to be in single directory, with each corresponding pair named as <filename>.ndpi and <filename>.ndpi.ndpa, respectively. Note that each NDPI file is very large (20-50 GB), and considerable storage space is necessary.

Utilities for interacting with raw NDPI and NDPA data can be found in the src/data folder, and information on their implementation and use can be found in [src/data/README.md](src/data/README.md).

In addition, two auxiliary files accompany the dataset:
- annotation_by_category.csv: Provides summary-level statistics of the number and type of annotations per annotated NDPI file.
- annotation_categories.csv: Provides a mapping of annotation names, which may include full species names, to six prediction classes.

### HDF5 Data Format
For efficient processing and analysis, the data is converted into HDF5 (.h5) files. Each HDF5 file contains multiple tiles (subregions) from the original images, along with relevant metadata and annotations.

- **Tiles:** Each tile is stored as a 4D array of shape (width, height, channels, focal plane), representing a small region of the original image across all color channels and focal planes.
- **Groups:** Each tile is a group in the HDF5 file, containing datasets for image data, bounding boxes, labels, and derived results (e.g., focus stacked, MIP).
- **Attributes:** Metadata such as tile coordinates, sample information, and processing parameters are stored as group or file attributes.

This structure enables fast access to image regions and supports downstream tasks such as focus stacking, maximum intensity projection, and machine learning workflows. Refer to the data preprocessing section for how this data is generated from raw NDPI and NDPA files.

### COCO Data Format
The data can additionally be exported into the standard COCO format, which facilitates model development. Exporting to this format does not retain all data stored in the H5 format, but can lead to improved training times due to increased compatibility with external model implementations as well as minimal overhead to decompress files, in contrast to the H5 format. Refer to the data preprocessing section for how this data is generated from  the H5 format.

## Data Preprocessing

The preprocessing pipeline used in this analysis contains multiple stages, including tiling the original NDPI files, generating focus stacked images, and exporting the data into the COCO format for training. More information on the implementation and how to execute the preprocessing pipeline can be found in the following README, found in [src/preprocessing/README.md](src/preprocessing/README.md).

## Exploratory Data Analysis

The src/exploration folder contains much of the exploratory data analysis performed in this project, including exploration of annotation density, the distribution of palynomorph subtypes, the sharpness of various focal planes in our multifocal data, and more. A detailed description of the available scripts and how to reproduce this analysis can be found in the following README, found in [src/exploration/README.md](src/exploration/README.md).

## Modeling & Evaluation

This repository currently contains implementations of YOLO26 and RF-DETR for palynomoprh detection. Standard object detection evaluation metrics including precision, recall, mAP@50, and mAP@50-95 are available. More information and training and evaluating these models can be found in the following README, found in [src/models/README.md](src/models/README.md).

## Usage

`src/annotator` serves the primary usage of this repository: providing an end-to-end pipeline for palynomorph annotation. [src/annotator/README.md](src/annotator/README.md) provides detailed instructions on running this pipeline.
