"""
Data I/O for NDPI whole-slide images and NDPA annotation files.

Modules:
    ndpi_reader — Load NDPI files, extract metadata, and read multi-focal-plane tiles
    ndpa_reader — Parse NDPA XML annotation files into structured Python objects
    ndpa_writer — Write bounding-box and circle annotations back to NDPA XML
    util        — Coordinate conversion between nanometer and pixel space
"""
