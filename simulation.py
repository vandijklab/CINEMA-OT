import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scsim_master import scsim
import pandas as pd

import random

def numbers_with_sum(n, k):
    """n numbers with sum k"""
    if n == 1:
        return [k]
    num = random.randint(0, k)
    return [num] + numbers_with_sum(n - 1, k - num)

random.seed(0)
np.random.seed(0)
for i in range(15):
    states_num = round(i/5) + 2
    gp = numbers_with_sum(states_num, 10-states_num)
    simulator = scsim.scsim(ngenes=1000, ncells=5000, seed = i, ngroups=states_num, libloc=7.64, libscale=0.78,
                 mean_rate=7.68,mean_shape=0.34, expoutprob=0.00286,
                 expoutloc=6.15, expoutscale=0.49,
                 diffexpprob=.5, diffexpdownprob=.5, diffexploc=1, diffexpscale=1,
                 bcv_dispersion=0.448, bcv_dof=22.087, ndoublets=0,groupprob=(np.array(gp)+1)/10,proggoups=[1,2],nproggenes=500,
                 progdeloc=1,progdescale=1,progdownprob=0.,progcellfrac = 1.)
    
    simulator.simulate()
    tmpobs = simulator.cellparams
    ## "Groups" represent the treatment variable
    tmpobs['Groups'] = 0
    tmpobs['Response_state'] = 0
    response_num = round(i/5) + 1
    attribution_matrix = np.zeros([states_num,response_num])
    simulator2_counts = simulator.counts.iloc[:,0:500].copy()
    for j in range(states_num):
        ncells_j = np.sum(simulator.cellparams['group']==(j+1))
        
        group = np.random.randint(0,2,size=ncells_j)
    
        gp2 = np.zeros(response_num+1) + 0.5
        gp2[1:] = (np.array(numbers_with_sum(response_num, 5)))/10
        
        simulator2 = scsim.scsim(ngenes=500, ncells=ncells_j, seed = 300, ngroups=response_num+1, libloc=7.64, libscale=0.78,
                     mean_rate=7.68,mean_shape=0.34, expoutprob=0.00286,
                     expoutloc=6.15, expoutscale=0.49,
                     diffexpprob=.5, diffexpdownprob=.5, diffexploc=1, diffexpscale=1,
                     bcv_dispersion=0.148, bcv_dof=22.087, ndoublets=0,groupprob=gp2,nproggenes=0,
                     progdeloc=1,progdescale=1,progdownprob=0.,progcellfrac = 1.)
        
        attribution_matrix[j,:] = 2 * gp2[1:]
        simulator2.simulate()
        ## group==1 is assigned as control, set the rest as perturbed
        tmpobs['Groups'][simulator.cellparams['group']==(j+1)] = (simulator2.cellparams['group'].values > 1) * 1 + 1
        tmpobs['Response_state'][simulator.cellparams['group']==(j+1)] = simulator2.cellparams['group'].values
        simulator2_counts.loc[simulator.cellparams['group']==(j+1),:] = simulator2.counts.values
    ## in the final anndata, 'group' represents cell state / type, 'Groups' represents treated or not, 'Response_state' indicates response heterogeneity
    adata = sc.AnnData(pd.concat([simulator.counts, simulator2_counts], axis=1),obs=tmpobs)
    adata.uns['attribution'] = attribution_matrix
    adata.write('ScsimBenchmarkData/adata'+str(i)+'.h5ad')