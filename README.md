# Fourier-Domain Epitaxial Thickness Inversion

A physics-informed frequency-domain method for semiconductor epitaxial layer thickness estimation based on infrared interference spectra.

## Overview

This project focuses on the non-destructive measurement of semiconductor epitaxial layer thickness from infrared reflectance spectra. By combining optical interference modeling and frequency-domain signal processing, the hidden thickness information is extracted from interference patterns in measured spectra.

The method transforms the thickness estimation problem into a frequency analysis problem:

**Reflectance Spectrum → Interference Frequency → Optical Path Difference → Epitaxial Thickness**

The project was developed for the National Undergraduate Mathematical Contest in Modeling (CUMCM) and received **Shanghai Second Prize**.

---

## Methodology

### 1. Optical Interference Modeling

- Established dual-beam interference models to describe the relationship between interference patterns and epitaxial layer thickness.
- Extended the model to multi-beam interference scenarios based on Fabry–Pérot theory.
- Derived the relationship between interference frequency and optical path difference, providing the theoretical basis for thickness inversion.

### 2. Frequency-Domain Thickness Estimation

A Fourier-transform-based pipeline was designed to extract thickness-related features from infrared reflectance spectra.

Main steps:

1. Spectral preprocessing
   - Data normalization
   - Trend removal
   - Noise reduction

2. Frequency feature extraction
   - Fast Fourier Transform (FFT)
   - Sliding-window spectral analysis
   - Peak frequency localization

3. Thickness inversion

The interference frequency extracted from FFT is converted into epitaxial thickness using the optical interference model.

---

## Handling Material Dispersion

The refractive index of semiconductor materials varies with optical wavelength (or wavenumber), which introduces calculation errors if treated as a constant.

To improve accuracy:

- Considered refractive index variation across different spectral regions.
- Applied local correction based on spectral position.
- Combined multiple estimation results using statistical methods to improve robustness.

The proposed method achieved a stable thickness estimation of approximately:

**SiC epitaxial layer thickness: ~7.9 μm**

---

## Multi-Beam Interference Analysis

For silicon samples, harmonic components in the frequency spectrum were analyzed to identify multi-beam interference effects.

By detecting characteristic frequency multiples:

- Evaluated the influence of higher-order interference.
- Verified the thickness estimation method under multi-beam interference conditions.

---

## Results

| Sample | Incident Angle | Estimated Thickness |
|------|------|------|
| SiC | 10° | ~7.92 μm |
| SiC | 15° | ~7.87 μm |
| Si | 10° | ~3.66 μm |
| Si | 15° | ~3.64 μm |

The consistency between different incident angles demonstrates the reliability of the proposed inversion method.
