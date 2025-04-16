# Causal INdependent Effect Module Attribution + Optimal Transport (CINEMA-OT)

CINEMA-OT is a **causal** framework for perturbation effect analysis to identify **individual treatment effects** and **synergy** at the **single cell** level. 

**Note**: Newer versions of CINEMA-OT are maintained at [Pertpy](https://github.com/scverse/pertpy).

## Architecture

<img width="1460" alt="image" src="https://user-images.githubusercontent.com/68533876/228745549-8328ea36-25c6-4665-9c68-bab1e1a78ef9.png">


Read our preprint on bioRxiv:

- Dong, Mingze, et al. "Causal identification of single-cell experimental perturbation effects with CINEMA-OT". bioRxiv (2022).
[https://www.biorxiv.org/content/10.1101/2022.07.31.502173v3](https://www.biorxiv.org/content/10.1101/2022.07.31.502173v3)

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
