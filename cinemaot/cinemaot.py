import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from . import sinkhorn_knopp as skp
from scipy.sparse import issparse

# Import Chatterjee score package from R
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

from sklearn.decomposition import FastICA
import sklearn.metrics
import meld


def cinemaot_unweighted(adata,obs_label,ref_label,expr_label,dim=20,thres=0.15,smoothness=1e-4,eps=1e-3,mode='parametric',marker=None):
    """
    Parameters
    ----------
    adata: 'AnnData'
        An anndata object containing the whole gene count matrix and an observation index for treatments. It should be preprocessed before input.
    obs_label: 'str'
        A string for indicating the treatment column name in adata.obs.
    ref_label: 'str'
        A string for indicating the control group in adata.obs.values.
    expr_label: 'str'
        A string for indicating the experiment group in adata.obs.values.
    dim: 'int'
        The number of independent components.
    thres: 'float'
        The threshold for setting the Chatterjee coefficent for confounder separation.
    smoothness: 'float'
        The parameter for setting the smoothness of entropy-regularized optimal transport. Should be set as a small value above zero!
    eps: 'float'
        The parameter for stop condition of OT convergence. 
    mode: 'str'
        If mode is 'parametric', return standard differential matrices. If it's non-parametric, we return expr cells' weighted quantile.
    Return
    ----------
    cf: 'numpy.ndarray'
        Confounder components, of shape (n_cells,n_components).
    ot: 'numpy.ndarray'
        Transport map across control and experimental conditions.
    te2: 'numpy.ndarray'
        Single-cell differential expression for each cell in control condition, of shape (n_refcells, n_genes).
    """
    transformer = FastICA(n_components=dim, random_state=0)
    X_transformed = transformer.fit_transform(adata.obsm['X_pca'][:,:dim])
    importr("XICOR")
    xicor = ro.r["xicor"]
    groupvec = (adata.obs[obs_label]==ref_label *1).values #control
    xi = np.zeros(dim)
    pval = np.zeros(dim)
    j = 0
    for source_row in X_transformed.T:
        rresults = xicor(ro.FloatVector(source_row), ro.FloatVector(groupvec), pvalue = True)
        xi[j] = np.array(rresults.rx2("xi"))[0]
        pval[j] = np.array(rresults.rx2("pval"))[0]
        j = j+1
    cf = X_transformed[:,xi<thres]
    cf1 = cf[adata.obs[obs_label]==expr_label,:] #expr
    cf2 = cf[adata.obs[obs_label]==ref_label,:] #control
    if sum(xi<thres)==1:
        dis = sklearn.metrics.pairwise_distances(cf1.reshape(-1,1),cf2.reshape(-1,1))
    elif sum(xi<thres)==0:
        raise ValueError("No confounder components identified. Please try a higher threshold.")
    else:
        dis = sklearn.metrics.pairwise_distances(cf1,cf2)
    e = smoothness * sum(xi<thres)
    af = np.exp(-dis * dis / e)
    r = np.zeros([cf1.shape[0],1])
    c = np.zeros([cf2.shape[0],1])
    r[:,0] = 1/cf1.shape[0]
    c[:,0] = 1/cf2.shape[0]
    sk = skp.SinkhornKnopp(setr=r,setc=c,epsilon=eps)
    ot = sk.fit(af).T
    if mode == 'parametric':
        if issparse(adata.X):
            te2 = adata.X.toarray()[adata.obs[obs_label]==ref_label,:] - np.matmul(ot/np.sum(ot,axis=1)[:,None],adata.X.toarray()[adata.obs[obs_label]==expr_label,:])
        else:
            te2 = adata.X[adata.obs[obs_label]==ref_label,:] - np.matmul(ot/np.sum(ot,axis=1)[:,None],adata.X[adata.obs[obs_label]==expr_label,:])
    elif mode == 'non_parametric':
        if issparse(adata.X):
            ref = adata.X.toarray()[adata.obs[obs_label]==ref_label,:]
            ref = ref[:,adata.var_names.isin(marker)]
            expr = adata.X.toarray()[adata.obs[obs_label]==expr_label,:]
            expr = expr[:,adata.var_names.isin(marker)]
            te2 = ref * 0
            for i in range(te2.shape[0]):
                te2[i,:] = weighted_quantile(expr,ref[i,:],sample_weight=ot[i,:])
        else:
            ref = adata.X[adata.obs[obs_label]==ref_label,:]
            ref = ref[:,adata.var_names.isin(marker)]
            expr = adata.X[adata.obs[obs_label]==expr_label,:]
            expr = expr[:,adata.var_names.isin(marker)]
            te2 = ref * 0
            for i in range(te2.shape[0]):
                te2[i,:] = weighted_quantile(expr,ref[i,:],sample_weight=ot[i,:])            
    else:
        raise ValueError("We do not support other methods for DE now.")
    return cf, ot, te2



def cinemaot_weighted(adata,obs_label,ref_label,expr_label,dim=20,thres=0.75,smoothness=1e-4,eps=1e-3,iter_num=2,k=3,mode='parametric',marker=None):
    """
    Parameters
    ----------
    adata: 'AnnData'
        An anndata object containing the whole gene count matrix and an observation index for treatments. It should be preprocessed before input.
    obs_label: 'str'
        A string for indicating the treatment column name in adata.obs.
    ref_label: 'str'
        A string for indicating the control group in adata.obs.values.
    expr_label: 'str'
        A string for indicating the experiment group in adata.obs.values.
    dim: 'int'
        The number of independent components.
    thres: 'float'
        The threshold for setting the Chatterjee coefficent for confounder separation.
    smoothness: 'float'
        The parameter for setting the smoothness of entropy-regularized optimal transport. Should be set as a small value above zero!
    eps: 'float'
        The parameter for stop condition of OT convergence. 
    iter_num: 'int'
        Iteration number for reweighting.
    k: 'int'
        The parameter for knn for MELD, in order to estimate propensity score.

    Return
    ----------
    cf: 'numpy.ndarray'
        Confounder components, of shape (n_cells,n_components).
    ot: 'numpy.ndarray'
        Transport map across control and experimental conditions.
    te2: 'numpy.ndarray'
        Single-cell differential expression for each cell in control condition, of shape (n_refcells, n_genes).
    r: 'numpy.ndarray'
        Propensity score weights for expr condition.
    c: 'numpy.ndarray'
        Propensity score weights for reference condition.       
    """
    sk = skp.SinkhornKnopp()
    data = adata.obsm['X_pca'][adata.obs[obs_label].isin([expr_label,ref_label]),:dim]
    #r = 20
    importr("XICOR")
    indexF = np.arange(data.shape[0])
    #iter_num = 2
    for i in range(iter_num):
        transformer = FastICA(n_components=dim, random_state=0)
        X_transformed = transformer.fit_transform(data)
        xicor = ro.r["xicor"]
        if i>0:
            groupvec = (adata.obs[obs_label]==ref_label *1).values[indexF]
        else:
            groupvec = (adata.obs[obs_label]==ref_label *1).values
        xi = np.zeros(dim)
        pval = np.zeros(dim)
        j = 0
        #thres = 0.75
        for source_row in X_transformed.T:
            rresults = xicor(ro.FloatVector(source_row), ro.FloatVector(groupvec), pvalue = True)
            xi[j] = np.array(rresults.rx2("xi"))[0]
            pval[j] = np.array(rresults.rx2("pval"))[0]
            j = j+1
        cf = X_transformed[:,xi<thres]
        tmp1 = (adata.obs[obs_label]==expr_label)
        tmp2 = (adata.obs[obs_label]==ref_label)
        cf1 = cf[tmp1[indexF],:]
        cf2 = cf[tmp2[indexF],:]
        #k = 3
        s = meld.MELD(knn=k).fit(cf)
        if i>0:
            smoothsig = s.transform(adata.obs[obs_label].values[indexF])
        else:
            smoothsig = s.transform(adata.obs[obs_label].values)
        r = np.zeros([cf1.shape[0],1])
        c = np.zeros([cf2.shape[0],1])
        r[:,0] = smoothsig.values[tmp1[indexF],1]/cf1.shape[0]
        c[:,0] = smoothsig.values[tmp2[indexF],0]/cf2.shape[0]
        weightvec = np.zeros(cf.shape[0])
        weightvec[tmp1[indexF]] = r[:,0]
        weightvec[tmp2[indexF]] = c[:,0]
        if i < (iter_num-1):
            np.random.seed(0)
            index_sample = np.random.choice(cf.shape[0],size = 3 * cf.shape[0], p = weightvec/np.sum(weightvec))
            indexF = indexF[index_sample]
            data = data[index_sample,:]
    
    X_transformed = transformer.transform(adata.obsm['X_pca'][adata.obs[obs_label].isin([expr_label,ref_label]),:dim])
    cf = X_transformed[:,xi<thres]
    cf1 = cf[tmp1,:]
    cf2 = cf[tmp2,:]
    s = meld.MELD(knn=k).fit(cf)
    smoothsig = s.transform(adata.obs[obs_label].values)
    r = np.zeros([cf1.shape[0],1])
    c = np.zeros([cf2.shape[0],1])
    r[:,0] = smoothsig.values[tmp1,1]/cf1.shape[0]
    c[:,0] = smoothsig.values[tmp2,0]/cf2.shape[0]
    if sum(xi<thres)==1:
        dis = sklearn.metrics.pairwise_distances(cf1.reshape(-1,1),cf2.reshape(-1,1))
    elif sum(xi<thres)==0:
        raise ValueError("No confounder components identified. Please try a higher threshold.")
    else:
        dis = sklearn.metrics.pairwise_distances(cf1,cf2)
    e = smoothness * sum(xi<thres)
    af = np.exp(-dis * dis / e)
    sk = skp.SinkhornKnopp(setr=r,setc=c,epsilon=eps)
    ot = sk.fit(af).T
    if mode == 'parametric':
        if issparse(adata.X):
            te2 = adata.X.toarray()[adata.obs[obs_label]==ref_label,:] - np.matmul(ot/np.sum(ot,axis=1)[:,None],adata.X.toarray()[adata.obs[obs_label]==expr_label,:])
        else:
            te2 = adata.X[adata.obs[obs_label]==ref_label,:] - np.matmul(ot/np.sum(ot,axis=1)[:,None],adata.X[adata.obs[obs_label]==expr_label,:])
    elif mode == 'non_parametric':
        if issparse(adata.X):
            ref = adata.X.toarray()[adata.obs[obs_label]==ref_label,:]
            ref = ref[:,adata.var_names.isin(marker)]
            expr = adata.X.toarray()[adata.obs[obs_label]==expr_label,:]
            expr = expr[:,adata.var_names.isin(marker)]
            te2 = ref * 0
            for i in range(te2.shape[0]):
                te2[i,:] = weighted_quantile(expr,ref[i,:],sample_weight=ot[i,:])
        else:
            ref = adata.X[adata.obs[obs_label]==ref_label,:]
            ref = ref[:,adata.var_names.isin(marker)]
            expr = adata.X[adata.obs[obs_label]==expr_label,:]
            expr = expr[:,adata.var_names.isin(marker)]
            te2 = ref * 0
            for i in range(te2.shape[0]):
                te2[i,:] = weighted_quantile(expr,ref[i,:],sample_weight=ot[i,:]) 
    else:
        raise ValueError("We do not support other methods for DE now.")
    return cf, ot, te2, r, c

def synergy(adata,obs_label,base,A,B,AB,dim=20,thres=0.15,smoothness=1e-4,eps=1e-3,mode='parametric'):
    adata1 = adata[adata.obs[obs_label].isin([base,A]),:]
    adata2 = adata[adata.obs[obs_label].isin([B,AB]),:]
    adata_link = adata[adata.obs[obs_label].isin([base,B]),:]
    cf, ot1, de1 = cinemaot_unweighted(adata1,obs_label=obs_label, ref_label=base, expr_label=A,dim=dim,thres=thres,smoothness=smoothness,eps=eps,mode=mode)
    cf, ot2, de2 = cinemaot_unweighted(adata2,obs_label=obs_label, ref_label=B, expr_label=AB,dim=dim,thres=thres,smoothness=smoothness,eps=eps,mode=mode)
    cf, ot0, de0 = cinemaot_unweighted(adata_link,obs_label=obs_label, ref_label=base, expr_label=B,dim=dim,thres=thres,smoothness=smoothness,eps=eps,mode=mode)
    if mode == 'parametric':
        syn = (ot0/np.sum(ot0,axis=1)[:,None]) @ de2 - de1
    elif mode == 'non_parametric':
        raise ValueError("We do not non-parametric synergy now.")
    else:
        raise ValueError("We do not support other methods for synergy now.")
    return syn


def weighted_quantile(values, num, sample_weight=None, 
                      values_sorted=False):
    """
    Estimate weighted quantile for robust estimation of gene expression change given the OT map. The function is completely vectorized to accelerate computation
    """
    values = np.array(values)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sorter = np.argsort(values,axis=0)
    values = np.take_along_axis(values, sorter, axis=0)
    sample_weight = np.tile(sample_weight/np.sum(sample_weight),(1,values.shape[1]))
    sample_weight = np.take_along_axis(sample_weight,sorter,axis=1)
    weighted_quantiles = np.cumsum(sample_weight,axis=0)
    weighted_quantiles = np.vstack((np.zeros(values.shape[1]),weighted_quantiles))
    numindex = np.sum(values <= num.reshape(1,-1),axis=0)
    return np.diag(weighted_quantiles[np.ix_(numindex,np.arange(values.shape[1]))])


def wgcna_module_scores(de_matrix, gene_names, n_variable_genes=2000):
    """
    Caculate gene modules and soft connectivity scores using WGCNA
    """
    wgcna = importr('WGCNA')
    variance = np.var(de_matrix, axis=0)
    genes_to_select = np.argsort(-variance) < n_variable_genes
    de_trimmed = de_matrix[:,genes_to_select]
    modules = wgcna.blockwiseModules(de_trimmed, numericLabels=True)
    # calculate top hub genes per module
    soft_connectivities = wgcna.softConnectivity(de_trimmed)
    return pd.DataFrame({
        'gene_name': gene_names[genes_to_select],
        'module': modules.rx2('colors').astype(int),
        'soft_connectivity': soft_connectivities
    })
