## Causal INdependent Effect Module Attribution + Optimal Transport (CINEMA-OT)

![Fig1](https://user-images.githubusercontent.com/68533876/159141178-f356d07e-21a3-4f79-9204-ec3d747035b2.png)

Causal INdependent Effect Module Attribution + Optimal Transport (CINEMA-OT)  is an algorithm for causal identification of single-cell level perturbation effects. The method separates confounder signals and treatment-associated signals based on modular gene regulation assumption. The method is robust to state-specific treatment effects with distributional matching. Also the method can tackle differential abundance across treatment conditions via iterative weighting. Differential abundance balance can also be performed based on pre-given cell type labels. 

### Dependency

- Python 3.7+
- Numpy
- Scanpy
- sklearn
- meld (for iterative weighting)
- rpy2
- XICOR (in R)

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

### Reference

