## Causal INdependent Effect Module Attribution + Optimal Transport (CINEMA-OT)

<img width="940" alt="image" src="https://user-images.githubusercontent.com/68533876/182231769-72c3395d-acff-4ad3-85ed-e137442ba6f0.png">


Causal INdependent Effect Module Attribution + Optimal Transport (CINEMA-OT)  is an algorithm for causal identification of single-cell level perturbation effects. The method separates confounder signals and treatment-associated signals based on modular gene regulation assumption. The method is robust to state-specific treatment effects with distributional matching. Also the method can tackle differential abundance across treatment conditions via either iterative weighting or pre-given cell type labels. 

### Dependency

- Python 3.7+
- Numpy
- Pandas
- Scanpy
- sklearn
- Scipy
- Meld (for iterative weighting)
- rpy2
- XICOR (in R)
- GSEAPY (for downstream gene set enrichment analysis)
- Scib (for benchmark)
- Harmonypy (for benchmark)

### Usage

```python
import numpy as np
import scanpy as sc
import cinemaot as co
adata = sc.read_h5ad('RealData/rvcse_220105.h5ad')
subset = adata[adata.obs['batch'].isin(['CSE','RVCSE']),:]
cf_unweighted, ot_unweighted, de_unweighted = co.cinemaot.cinemaot_unweighted(subset,obs_label='batch', ref_label='CSE', expr_label='RVCSE')
cf_weighted, ot_weighted, de_weighted, r_weighted, c_weighted = co.cinemaot.cinemaot_weighted(subset,obs_label='batch', ref_label='CSE', expr_label='RVCSE')
```

For a more detailed tutorial, see cinemaot_tutorial.ipynb. The data used in the tutorial can be accessed at: https://drive.google.com/file/d/1A3rNdgfiXFWhCUOoUfJ-AiY7AAOU0Ie3/view?usp=sharing

### Reference

