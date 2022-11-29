import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from . import sinkhorn_knopp as skp
from . import utils
from scipy.sparse import issparse
# In this newer version we use the Python implementation of xicor
# import rpy2.robjects as ro
# import rpy2.robjects.numpy2ri
# import rpy2.robjects.pandas2ri
# from rpy2.robjects.packages import importr
# rpy2.robjects.numpy2ri.activate()
# rpy2.robjects.pandas2ri.activate()


# In order to estimate a sharper OT matrix, we provide an option "ot.bregman.sinkhorn_epsilon_scaling" instead of original sinkhorn
# Instead of projecting the whole count matrix, we use the pca result of projected ICA components to stablize the noise
# returning an anndata object
# Detecting differently expressed genes: G = A + Z + AZ + e by NB regression. Significant coefficient before AZ means conditional-specific effect
# Further exclusion of false positives may be removed by permutation (as in PseudotimeDE)

from xicor.xicor import Xi
import ot

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import FastICA
import sklearn.metrics
import meld


def cinemaot_unweighted(adata,obs_label,ref_label,expr_label,dim=20,thres=0.15,smoothness=1e-4,eps=1e-3,mode='parametric',ot_setting='original',marker=None,preweight_label=None):
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
    if dim is None:
        sk = skp.SinkhornKnopp()
        c = 0.5
        data=adata.X
        vm = (1e-3 + data + c * data * data)/(1+c)
        P = sk.fit(vm)
        wm = np.dot(np.dot(np.sqrt(sk._D1),vm),np.sqrt(sk._D2))
        u,s,vt = np.linalg.svd(wm)
        dim = np.min(sum(s > (np.sqrt(data.shape[0])+np.sqrt(data.shape[1]))),adata.obsm['X_pca'].shape[1])


    transformer = FastICA(n_components=dim, random_state=0)
    X_transformed = transformer.fit_transform(adata.obsm['X_pca'][:,:dim])
    #importr("XICOR")
    #xicor = ro.r["xicor"]
    groupvec = (adata.obs[obs_label]==ref_label *1).values #control
    xi = np.zeros(dim)
    #pval = np.zeros(dim)
    j = 0
    for source_row in X_transformed.T:
        xi_obj = Xi(source_row,groupvec*1)
        #rresults = xicor(ro.FloatVector(source_row), ro.FloatVector(groupvec), pvalue = True)
        #xi[j] = np.array(rresults.rx2("xi"))[0]
        xi[j] = xi_obj.correlation
        #pval[j] = np.array(rresults.rx2("pval"))[0]
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
    if preweight_label is None:
        r[:,0] = 1/cf1.shape[0]
        c[:,0] = 1/cf2.shape[0]
    else:
        #implement a simple function here, taking adata.obs, output inverse prob weight. For consistency, c is still the empirical distribution, while r is weighted.
        adata1 = adata[adata.obs[obs_label]==expr_label,:]
        adata2 = adata[adata.obs[obs_label]==ref_label,:]
        c[:,0] = 1/cf2.shape[0]
        for ct in list(set(adata1.obs[preweight_label].values.tolist())):
            r[(adata1.obs[preweight_label]==ct).values,0] = np.sum((adata2.obs[preweight_label]==ct).values) / np.sum((adata1.obs[preweight_label]==ct).values)
        r[:,0] = r[:,0]/np.sum(r[:,0])

    if ot_setting == 'original':
        sk = skp.SinkhornKnopp(setr=r,setc=c,epsilon=eps)
        ot_matrix = sk.fit(af).T

    elif ot_setting == 'bregman':
        ot_matrix = ot.bregman.sinkhorn_epsilon_scaling(r[:,0],c[:,0],dis,e,epsilon0=1,stopThr=eps)

    embedding = X_transformed[adata.obs[obs_label]==ref_label,:] - np.matmul(ot_matrix/np.sum(ot_matrix,axis=1)[:,None],X_transformed[adata.obs[obs_label]==expr_label,:])

    if mode == 'parametric':
        if issparse(adata.X):
            te2 = adata.X.toarray()[adata.obs[obs_label]==ref_label,:] - np.matmul(ot_matrix/np.sum(ot_matrix,axis=1)[:,None],adata.X.toarray()[adata.obs[obs_label]==expr_label,:])
        else:
            te2 = adata.X[adata.obs[obs_label]==ref_label,:] - np.matmul(ot_matrix/np.sum(ot_matrix,axis=1)[:,None],adata.X[adata.obs[obs_label]==expr_label,:])
    elif mode == 'non_parametric':
        if issparse(adata.X):
            ref = adata.X.toarray()[adata.obs[obs_label]==ref_label,:]
            ref = ref[:,adata.var_names.isin(marker)]
            expr = adata.X.toarray()[adata.obs[obs_label]==expr_label,:]
            expr = expr[:,adata.var_names.isin(marker)]
            te2 = ref * 0
            for i in range(te2.shape[0]):
                te2[i,:] = weighted_quantile(expr,ref[i,:],sample_weight=ot_matrix[i,:])
        else:
            ref = adata.X[adata.obs[obs_label]==ref_label,:]
            ref = ref[:,adata.var_names.isin(marker)]
            expr = adata.X[adata.obs[obs_label]==expr_label,:]
            expr = expr[:,adata.var_names.isin(marker)]
            te2 = ref * 0
            for i in range(te2.shape[0]):
                te2[i,:] = weighted_quantile(expr,ref[i,:],sample_weight=ot_matrix[i,:])            
    else:
        raise ValueError("We do not support other methods for DE now.")

    TE = sc.AnnData(te2,obs=adata[adata.obs[obs_label]==ref_label,:].obs.copy(),var=adata.var.copy())
    TE.obsm['X_embedding'] = embedding
    return cf, ot_matrix, TE



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
    if dim is None:
        sk = skp.SinkhornKnopp()
        c = 0.5
        data=adata.X
        vm = (1e-3 + data + c * data * data)/(1+c)
        P = sk.fit(vm)
        wm = np.dot(np.dot(np.sqrt(sk._D1),vm),np.sqrt(sk._D2))
        u,s,vt = np.linalg.svd(wm)
        dim = sum(s > (np.sqrt(data.shape[0])+np.sqrt(data.shape[1])))

    sk = skp.SinkhornKnopp()
    data = adata.obsm['X_pca'][adata.obs[obs_label].isin([expr_label,ref_label]),:dim]
    #r = 20
    #importr("XICOR")
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
        #pval = np.zeros(dim)
        j = 0
        #thres = 0.75
        for source_row in X_transformed.T:
            xi_obj = Xi(source_row,groupvec*1)
            #rresults = xicor(ro.FloatVector(source_row), ro.FloatVector(groupvec), pvalue = True)
            #xi[j] = np.array(rresults.rx2("xi"))[0]
            xi[j] = xi_obj.correlation
            #pval[j] = np.array(rresults.rx2("pval"))[0]
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

def synergy(adata,obs_label,base,A,B,AB,dim=20,thres=0.15,smoothness=1e-4,eps=1e-3,mode='parametric',preweight_label=None,path=None,fthres=None):
    adata1 = adata[adata.obs[obs_label].isin([base,A]),:]
    adata2 = adata[adata.obs[obs_label].isin([B,AB]),:]
    adata_link = adata[adata.obs[obs_label].isin([base,B]),:]
    cf, ot1, de1 = cinemaot_unweighted(adata1,obs_label=obs_label, ref_label=base, expr_label=A,dim=dim,thres=thres,smoothness=smoothness,eps=eps,mode='parametric',preweight_label=preweight_label)
    cf, ot2, de2 = cinemaot_unweighted(adata2,obs_label=obs_label, ref_label=B, expr_label=AB,dim=dim,thres=thres,smoothness=smoothness,eps=eps,mode='parametric',preweight_label=preweight_label)
    cf, ot0, de0 = cinemaot_unweighted(adata_link,obs_label=obs_label, ref_label=base, expr_label=B,dim=dim,thres=thres,smoothness=smoothness,eps=eps,mode='parametric',preweight_label=preweight_label)
    if mode == 'parametric':
        syn = sc.AnnData(-((ot0/np.sum(ot0,axis=1)[:,None]) @ de2.X - de1.X),obs=de1.obs.copy(),var=de1.var.copy())
        syn.obsm['X_embedding'] = (ot0/np.sum(ot0,axis=1)[:,None]) @ de2.obsm['X_embedding'] - de1.obsm['X_embedding']
        return syn
    elif mode == 'non_parametric':
        # For data with varying batch effect across conditions, we recommend output the difference set of significant genes
        syn2 = -(ot0/np.sum(ot0,axis=1)[:,None]) @ de2
        syn1 = -de1
        subset = adata[adata.obs[obs_label].isin([base]),:]
        syn2 = sc.AnnData(syn2)
        syn2.obs[preweight_label] = subset.obs[preweight_label].values
        syn2.var_names = subset.var_names
        syn1 = sc.AnnData(syn1)
        syn1.obs[preweight_label] = subset.obs[preweight_label].values
        syn1.var_names = subset.var_names
        utils.clustertest_synergy(syn1,syn2,preweight_label,1e-5,fthres,path=path)
        return
        #raise ValueError("We do not non-parametric synergy now.")
    else:
        raise ValueError("We do not support other methods for synergy now.")
        return


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

# For two conditions
def NBregression(adata,n_cells=200):
    cf = adata.obsm['cf']
    z = np.zeros([adata.shape[0],2])
    z[:,0] = 1
    z[:,1] = (np.array(adata.obs['perturbation'].values) == adata.obs['perturbation'].values[0]) * 1
    X = np.hstack((cf,z,cf*z[:,1][:,None]))
    effectsize = np.zeros(adata.raw.X.shape[1])
    pvalue = np.zeros(adata.raw.X.shape[1])
    for i in range(adata.raw.X.shape[1]):
        if np.sum(adata.raw.X[:,i].toarray()[:,0]>0)>=n_cells:
            glm_binom = sm.GLM(adata.raw.X[:,i].toarray()[:,0], X, family=sm.families.Poisson())
            try:
                res = glm_binom.fit(tol=1e-4)
                pvalue[i] = np.min(res.pvalues[cf.shape[1]+2:])
                effectsize[i] = res.params[np.argmin(res.pvalues[cf.shape[1]+2:])]
            except:
                pvalue[i] = 0
                effectsize[i] = 0

    return effectsize, pvalue


def attribution_scatter(adata,obs_label,control_label,expr_label,use_raw=True):
    cf = adata.obsm['cf']
    if use_raw:
        Y0 = adata.raw.X.toarray()[adata.obs[obs_label]==control_label,:]
        Y1 = adata.raw.X.toarray()[adata.obs[obs_label]==expr_label,:]
    else:
        Y0 = adata.X.toarray()[adata.obs[obs_label]==control_label,:]
        Y1 = adata.X.toarray()[adata.obs[obs_label]==expr_label,:]
    X0 = cf[adata.obs[obs_label]==control_label,:]
    X1 = cf[adata.obs[obs_label]==expr_label,:]
    ols0 = LinearRegression()
    ols0.fit(X0,Y0)
    ols1 = LinearRegression()
    ols1.fit(X1,Y1)
    c0 = ols0.predict(X0) - np.mean(ols0.predict(X0),axis=0)
    c1 = ols1.predict(X1) - np.mean(ols1.predict(X1),axis=0)
    e0 = Y0 - ols0.predict(X0)
    e1 = Y1 - ols1.predict(X1)
    #c_effect = np.mean(np.abs(ols1.coef_)+1e-6,axis=1) / np.mean(np.abs(ols0.coef_)+1e-6,axis=1)
    c_effect = (np.linalg.norm(c1,axis=0)+1e-6)/(np.linalg.norm(c0,axis=0)+1e-6)
    s_effect = (np.linalg.norm(e1,axis=0)+1e-6)/(np.linalg.norm(e0,axis=0)+1e-6)
    return c_effect, s_effect

