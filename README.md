# MorphTrace

## Digital Image Forensics & Manipulation Detection

MorphTrace is a web-based digital image forensics application designed
to analyze images using multiple forensic techniques and provide
explainable evidence about potential manipulation.

## Core Objectives

- Analyze image metadata and EXIF information
- Perform Error Level Analysis (ELA)
- Detect potential copy-move patterns
- Analyze compression inconsistencies
- Analyze noise inconsistencies
- Localize suspicious evidence where technically possible
- Aggregate forensic evidence
- Present an explainable manipulation assessment

## Evidence Localization

MorphTrace is designed to provide more than an image-level result.

When a forensic technique produces spatially meaningful evidence, the
system should identify and visualize the corresponding suspicious
region.

The system must distinguish between forensic evidence and certainty.
No individual technique should automatically be treated as definitive
proof of manipulation.

## Technology Stack

- Python
- Flask
- Pillow
- NumPy
- OpenCV
- HTML
- CSS
- JavaScript
- SQLite
- Machine learning components where appropriate
