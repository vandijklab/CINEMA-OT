import numpy as np
import scanpy as sc
from anndata import AnnData
from . import sinkhorn_knopp as skp

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


def cinemaot_unweighted(adata,obs_label,ref_label,expr_label,dim=20,thres=0.15,smoothness=1e-4,eps=1e-3):
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
    te2 = adata.X.toarray()[adata.obs[obs_label]==ref_label,:] - np.matmul(ot/np.sum(ot,axis=1)[:,None],adata.X.toarray()[adata.obs[obs_label]==expr_label,:])
    return cf, ot, te2



def cinemaot_weighted(adata,obs_label,ref_label,expr_label,dim=20,thres=0.75,smoothness=1e-4,eps=1e-3,iter_num=2,k=3):
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
    te2 = adata.X.toarray()[adata.obs[obs_label]==ref_label,:] - np.matmul(ot/np.sum(ot,axis=1)[:,None],adata.X.toarray()[adata.obs[obs_label]==expr_label,:])
    return cf, ot, te2, r, c
