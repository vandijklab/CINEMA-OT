import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from . import sinkhorn_knopp as skp
#from . import utils
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
import scipy.stats as ss

# In this newer version we use the Python implementation of xicor
# import rpy2.robjects as ro
# import rpy2.robjects.numpy2ri
# import rpy2.robjects.pandas2ri
# from rpy2.robjects.packages import importr
# rpy2.robjects.numpy2ri.activate()
# rpy2.robjects.pandas2ri.activate()


# Instead of projecting the whole count matrix, we use the pca result of projected ICA components to stablize the noise
# returning an anndata object
# Detecting differently expressed genes: G = A + Z + AZ + e by NB regression. Significant coefficient before AZ means conditional-specific effect
# Further exclusion of false positives may be removed by permutation (as in PseudotimeDE)

#import ot

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import FastICA
import sklearn.metrics


def cinemaot_unweighted(adata,obs_label,ref_label,expr_label,dim=20,thres=0.15,smoothness=1e-4,eps=1e-3,mode='parametric',marker=None,preweight_label=None):
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


    transformer = FastICA(n_components=dim, random_state=0,whiten="arbitrary-variance")
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

    sk = skp.SinkhornKnopp(setr=r,setc=c,epsilon=eps)
    ot_matrix = sk.fit(af).T

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



def cinemaot_weighted(adata,obs_label,ref_label,expr_label,use_rep=None,dim=20,thres=0.75,smoothness=1e-4,eps=1e-3,k=10,resolution=1,mode='parametric',marker=None):
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
        The parameter for knn.

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
    adata_ = adata[adata.obs[obs_label].isin([expr_label,ref_label])].copy()
    if use_rep is None:
        X_pca1 = adata_.obsm['X_pca'][adata_.obs[obs_label]==expr_label,:]
        X_pca2 = adata_.obsm['X_pca'][adata_.obs[obs_label]==ref_label,:]
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_pca1)
        mixscape_pca = adata.obsm['X_pca'].copy()
        mixscapematrix = nbrs.kneighbors_graph(X_pca2).toarray()
        mixscape_pca[adata_.obs[obs_label]==ref_label,:] = np.dot(mixscapematrix, mixscape_pca[adata_.obs[obs_label]==expr_label,:])/k

        adata_.obsm['X_mpca'] = mixscape_pca
        sc.pp.neighbors(adata_,use_rep='X_mpca')

    else:
        sc.pp.neighbors(adata_,use_rep=use_rep)
    sc.tl.leiden(adata_,resolution=resolution)

    z = np.zeros(adata_.shape[0]) + 1

    j = 0

    for i in adata_.obs['leiden'].cat.categories:
        if adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)].shape[0] >= adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)].shape[0]:
            z[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)] = adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)].shape[0] / adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)].shape[0]
            if j == 0:
                idx = sc.pp.subsample(adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)],n_obs = adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)].shape[0],copy=True).obs.index
                idx = idx.append(adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)].obs.index)
                j = j + 1
            else:
                idx_tmp = sc.pp.subsample(adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)],n_obs = adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)].shape[0],copy=True).obs.index
                idx_tmp = idx_tmp.append(adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)].obs.index)
                idx = idx.append(idx_tmp)
        else:
            z[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)] = adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)].shape[0] / adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)].shape[0]
            if j == 0:
                idx = sc.pp.subsample(adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)],n_obs = adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)].shape[0],copy=True).obs.index
                idx = idx.append(adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)].obs.index)
                j = j + 1
            else:
                idx_tmp = sc.pp.subsample(adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==expr_label)],n_obs = adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)].shape[0],copy=True).obs.index
                idx_tmp = idx_tmp.append(adata_[(adata_.obs['leiden']==i) & (adata_.obs[obs_label]==ref_label)].obs.index)
                idx = idx.append(idx_tmp)

    transformer = FastICA(n_components=dim, random_state=0, whiten="arbitrary-variance")
    X_transformed = transformer.fit_transform(adata_[idx].obsm['X_pca'][:,:dim])
    #importr("XICOR")
    #xicor = ro.r["xicor"]
    groupvec = (adata_[idx].obs[obs_label]==ref_label *1).values #control
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

    cf = transformer.transform(adata_.obsm['X_pca'][:,:dim])[:,xi<thres]

    cf1 = X_transformed[adata_[idx].obs[obs_label]==expr_label,:][:,xi<thres]
    cf2 = cf[adata_.obs[obs_label]==ref_label,:]
    r = np.zeros([cf1.shape[0],1])
    c = np.zeros([cf2.shape[0],1])
    r[:,0] = 1/cf1.shape[0]
    c[:,0] = 1/cf2.shape[0]
    #return cf,xi,adata_[idx]
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
    #return (adata.X[idx][adata[idx].obs[obs_label]==expr_label,:])

    if mode == 'parametric':
        cf[adata.obs[obs_label]==ref_label,:] = (ot/np.sum(ot,axis=1)[:,None]) @ (cf1)
        if issparse(adata.X):
            te2 = adata.X.toarray()[adata.obs[obs_label]==ref_label,:] - (ot/np.sum(ot,axis=1)[:,None]) @ (adata[idx].X.toarray()[adata[idx].obs[obs_label]==expr_label,:])
        else:
            te2 = adata.X[adata.obs[obs_label]==ref_label,:] - (ot/np.sum(ot,axis=1)[:,None]) @ (adata[idx].X[adata[idx].obs[obs_label]==expr_label,:])
        te2 = sc.AnnData(te2,obs=adata[adata.obs[obs_label]==ref_label,:].obs.copy(),var=adata.var.copy())
        embedding = transformer.transform(adata_.obsm['X_pca'][:,:dim])[adata_.obs[obs_label]==ref_label,:] - (ot/np.sum(ot,axis=1)[:,None]) @ (transformer.transform(adata_[idx].obsm['X_pca'][:,:dim])[adata_[idx].obs[obs_label]==expr_label,:])
        te2.obsm['X_embedding'] = embedding
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
    return cf, ot, te2, z[adata.obs[obs_label]==ref_label]

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
    #elif mode == 'non_parametric':
        # For data with varying batch effect across conditions, we recommend output the difference set of significant genes
        #syn2 = -(ot0/np.sum(ot0,axis=1)[:,None]) @ de2
        #syn1 = -de1
        #subset = adata[adata.obs[obs_label].isin([base]),:]
        #syn2 = sc.AnnData(syn2)
        #syn2.obs[preweight_label] = subset.obs[preweight_label].values
        #syn2.var_names = subset.var_names
        #syn1 = sc.AnnData(syn1)
        #syn1.obs[preweight_label] = subset.obs[preweight_label].values
        #syn1.var_names = subset.var_names
        #utils.clustertest_synergy(syn1,syn2,preweight_label,1e-5,fthres,path=path)
        #return
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


class Xi:
    """
    x and y are the data vectors
    """

    def __init__(self, x, y):

        self.x = x
        self.y = y

    @property
    def sample_size(self):
        return len(self.x)

    @property
    def x_ordered_rank(self):
        # PI is the rank vector for x, with ties broken at random
        # Not mine: source (https://stackoverflow.com/a/47430384/1628971)
        # random shuffling of the data - reason to use random.choice is that
        # pd.sample(frac=1) uses the same randomizing algorithm
        len_x = len(self.x)
        randomized_indices = np.random.choice(np.arange(len_x), len_x, replace=False)
        randomized = [self.x[idx] for idx in randomized_indices]
        # same as pandas rank method 'first'
        rankdata = ss.rankdata(randomized, method="ordinal")
        # Reindexing based on pairs of indices before and after
        unrandomized = [
            rankdata[j] for i, j in sorted(zip(randomized_indices, range(len_x)))
        ]
        return unrandomized

    @property
    def y_rank_max(self):
        # f[i] is number of j s.t. y[j] <= y[i], divided by n.
        return ss.rankdata(self.y, method="max") / self.sample_size

    @property
    def g(self):
        # g[i] is number of j s.t. y[j] >= y[i], divided by n.
        return ss.rankdata([-i for i in self.y], method="max") / self.sample_size

    @property
    def x_ordered(self):
        # order of the x's, ties broken at random.
        return np.argsort(self.x_ordered_rank)

    @property
    def x_rank_max_ordered(self):
        x_ordered_result = self.x_ordered
        y_rank_max_result = self.y_rank_max
        # Rearrange f according to ord.
        return [y_rank_max_result[i] for i in x_ordered_result]

    @property
    def mean_absolute(self):
        x1 = self.x_rank_max_ordered[0 : (self.sample_size - 1)]
        x2 = self.x_rank_max_ordered[1 : self.sample_size]
        
        return (
            np.mean(
                np.abs(
                    [
                        x - y
                        for x, y in zip(
                            x1,
                            x2,
                        )
                    ]
                )
            )
            * (self.sample_size - 1)
            / (2 * self.sample_size)
        )

    @property
    def inverse_g_mean(self):
        gvalue = self.g
        return np.mean(gvalue * (1 - gvalue))

    @property
    def correlation(self):
        """xi correlation"""
        return 1 - self.mean_absolute / self.inverse_g_mean

    @classmethod
    def xi(cls, x, y):
        return cls(x, y)

    def pval_asymptotic(self, ties=False, nperm=1000):
        """
        Returns p values of the correlation
        Args:
            ties: boolean
                If ties is true, the algorithm assumes that the data has ties
                and employs the more elaborated theory for calculating
                the P-value. Otherwise, it uses the simpler theory. There is
                no harm in setting tiles True, even if there are no ties.
            nperm: int
                The number of permutations for the permutation test, if needed.
                default 1000
        Returns:
            p value
        """
        # If there are no ties, return xi and theoretical P-value:

        if ties:
            return 1 - ss.norm.cdf(
                np.sqrt(self.sample_size) * self.correlation / np.sqrt(2 / 5)
            )

        # If there are ties, and the theoretical method
        # is to be used for calculation P-values:
        # The following steps calculate the theoretical variance
        # in the presence of ties:
        sorted_ordered_x_rank = sorted(self.x_rank_max_ordered)

        ind = [i + 1 for i in range(self.sample_size)]
        ind2 = [2 * self.sample_size - 2 * ind[i - 1] + 1 for i in ind]

        a = (
            np.mean([i * j * j for i, j in zip(ind2, sorted_ordered_x_rank)])
            / self.sample_size
        )

        c = (
            np.mean([i * j for i, j in zip(ind2, sorted_ordered_x_rank)])
            / self.sample_size
        )

        cq = np.cumsum(sorted_ordered_x_rank)

        m = [
            (i + (self.sample_size - j) * k) / self.sample_size
            for i, j, k in zip(cq, ind, sorted_ordered_x_rank)
        ]

        b = np.mean([np.square(i) for i in m])
        v = (a - 2 * b + np.square(c)) / np.square(self.inverse_g_mean)

        return 1 - ss.norm.cdf(
            np.sqrt(self.sample_size) * self.correlation / np.sqrt(v)
        )