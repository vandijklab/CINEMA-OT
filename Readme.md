# Causal INdependent Effect Module Attribution + Optimal Transport (CINEMA-OT)

CINEMA-OT is a **causal** framework for perturbation effect analysis to identify **individual treatment effects** and **synergy** at the **single cell** level.

## Architecture

<img width="1411" alt="CINEMAOT_gitfig" src="https://user-images.githubusercontent.com/68533876/204625392-f4de2fd1-8cd0-4aac-81f8-6155b52a7630.png">


Read our preprint on bioRxiv:

- Dong, Mingze, et al. "Causal identification of single-cell experimental perturbation effects with CINEMA-OT". bioRxiv (2022).
[https://www.biorxiv.org/content/10.1101/2022.07.31.502173v2](https://www.biorxiv.org/content/10.1101/2022.07.31.502173v2)

## System requirements
### Hardware requirements
`CINEMA-OT` requires only a standard computer with enough RAM to perform in-memory computations.
### OS requirements
The `CINEMA-OT` package is supported for all OS in principle. The package has been tested on the following systems:
* macOS: Monterey (12.4)
* Linux: RHEL Maipo (7.9), Ubantu (18.04)
### Dependencies
See `setup.cfg` for details.

## Installation
CINEMA-OT requires `python` version 3.7+.  Install directly from pip with:

    pip install cinemaot

The installation should take no more than a few minutes on a normal desktop computer.


## Usage

For detailed usage, follow our step-by-step tutorial here:

- [Getting Started with CINEMA-OT](https://github.com/vandijklab/CINEMA-OT/blob/main/cinemaot_tutorial.ipynb)

Download the data used for the tutorial here:

- [Ex vivo stimulation of human peripheral blood mononuclear cells (PBMC) with interferon](https://drive.google.com/file/d/1A3rNdgfiXFWhCUOoUfJ-AiY7AAOU0Ie3/view?usp=sharing)
