# Adaptive Learning Models

This repository contains the source code accompanying the paper *Adaptive learning via BG-thalamo-cortical circuitry*. It provides Python implementations of several models collectively referred to as LEIA (Learning as Entropy-Induced Attractor switch model).

## Repository Layout

- **Basic LEIA Model** – Python implementation of a baseline changepoint model.
- **LEIA with adaptive threshold** – Extended model in which the threshold for state transitions adapts over time.
- **LEIA with correlated noise** – Version allowing correlated variability across trials.
- **figure 1 - 6** – Scripts for generating plots.

## Requirements

The Python scripts were tested with Python&nbsp;3.8 and rely on `numpy`, `scipy` and `matplotlib`. Analyses inside `figure 3/figure3C_3D` additionally require a MATLAB environment.

## License

This project is distributed under the terms of the MIT License. See the `LICENSE` file for full details.