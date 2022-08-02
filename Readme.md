## Causal INdependent Effect Module Attribution + Optimal Transport (CINEMA-OT)

<img width="940" alt="image" src="https://user-images.githubusercontent.com/68533876/182231769-72c3395d-acff-4ad3-85ed-e137442ba6f0.png">

*Causal* framework for perturbation effect analysis to identify *individual treatment effects* and *synergy* at the *single cell* level.

Read our preprint on BioRXiv:

- Dong, Mingze, et al. "Causal identification of single-cell experimental perturbation effects with CINEMA-OT". bioRxiv (2022).
https://www.biorxiv.org/content/10.1101/2022.07.31.502173v1

# Installation
CINEMA-OT requires `python` version 3.7+.  Install directly from github with:

    pip install git+https://github.com/vandijklab/CINEMA-OT

## Dependencies
### Python
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)
- [scanpy](https://pypi.org/project/scanpy/)
- [sklearn](https://scikit-learn.org/stable/install.html)
- [scipy](https://pypi.org/project/scipy/)
- [meld](https://pypi.org/project/meld/)
- [rpy2](https://pypi.org/project/rpy2/)
- [gseapy](https://pypi.org/project/gseapy/)

### R
- [XICOR](https://cran.r-project.org/web/packages/XICOR/index.html)

### Usage

For detailed usage, follow our step-by-step tutorial here:

- [Getting Started with CINEMA-OT](https://github.com/vandijklab/CINEMA-OT/blob/main/cinemaot_tutorial.ipynb)

Download the data used for the tutorial here:

- [Ex vivo stimulation of human peripheral blood mononuclear cells (PBMC) with interferon](https://drive.google.com/file/d/1A3rNdgfiXFWhCUOoUfJ-AiY7AAOU0Ie3/view?usp=sharing)
