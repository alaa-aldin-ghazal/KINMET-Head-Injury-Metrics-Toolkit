# KINMET: Head Injury Metrics Toolkit

KINMET is an open-source, research-oriented Python software tool with a graphical
user interface (GUI) for computing head injury metrics from six-degree-of-freedom
(6-DOF) kinematic data. The software provides integrated post-processing
workflow for biomechanics, automotive safety, sports biomechanics, and related
research applications.

## Overview

The transformation of raw head kinematic data into injury metrics
typically involves multiple steps, including signal conditioning, filtering,
numerical integration, metric computation, and documentation. These steps are
often performed using fragmented scripts or proprietary software, which can
reduce transparency and reproducibility.

KINMET consolidates these steps into a single, accessible GUI-based application
that enables consistent, repeatable, and well-documented head-impact analysis.

## Key Features

- Interactive column mapping and unit selection (time, acceleration, angular velocity)
- Live preview of raw and filtered signals during data import
- Sampling: aware signal filtering with advisory feedback:
  - CFC filtering (e.g., CFC 60, CFC 180, CFC 600, CFC 1000)  
  - Butterworth low-pass filtering (configurable cutoff and order)
  - Savitzky–Golay smoothing
  - Moving-average filtering
- Automated computation of widely used head injury metrics:
  - Head Injury Criterion (HIC15, HIC36)
  - Brain Injury Criterion (BrIC)
  - Gadd Severity Index (SI)
  - Peak resultant linear acceleration and angular velocity
  - Estimated neck forces, moments, and Neck Injury Criterion (Nij)
- Automated generation of a multi-page PDF report, including:
  - Summary tables
  - Time-series plots
  - Highlighted HIC windows
  - BrIC and probability curves
  - Neck load histories
  - Mathematical definitions of injury metrics
- Export of computed metrics and full processing metadata to a structured JSON file
  for reproducibility and independent verification

## Intended Use and Disclaimer

KINMET is intended for **research and educational use only**.

The computed injury metrics, probability estimates, neck load calculations, and
Nij values are **not clinically validated** and must **not** be used for medical
diagnosis, regulatory certification, safety compliance decisions, or real-time
injury prediction. Results should be interpreted as comparative biomechanical
indicators within a controlled research context.

## Installation

### Requirements
- Python 3.8 or later
- Required Python packages:
  - numpy
  - scipy
  - pandas
  - matplotlib
  - tkinter

### Installation Steps

```bash
git clone https://github.com/<your-username>/KINMET.git
cd KINMET
pip install -r requirements.txt
