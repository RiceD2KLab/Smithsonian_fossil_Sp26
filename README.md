# Smithsonian_fossil_Sp26

## Table of Contents

- [Introduction](#introduction)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)

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

## Repo Structure

## Installation

### Prerequisites
- Python >= 3.9

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
The original dataset consists of multi-focal microscopy images. Each sample is stored as an NDPI image file, which contains images of the sample at various magnifications and focal planes. Associated annotations are provided in separate NDPA files, denoting the location of palynomorphs within the NDPI files.s

### HDF5 Data Format
For efficient processing and analysis, the data is converted into HDF5 (.h5) files. Each HDF5 file contains multiple tiles (subregions) from the original images, along with relevant metadata and annotations.

- **Tiles:** Each tile is stored as a 4D array of shape (width, height, channels, focal plane), representing a small region of the original image across all color channels and focal planes.
- **Groups:** Each tile is a group in the HDF5 file, containing datasets for image data, bounding boxes, labels, and derived results (e.g., focus stacked, MIP).
- **Attributes:** Metadata such as tile coordinates, sample information, and processing parameters are stored as group or file attributes.

This structure enables fast access to image regions and supports downstream tasks such as focus stacking, maximum intensity projection, and machine learning workflows.

## Exploratory Data Analysis

## Data Preprocessing

## Modeling

## Evaluation
