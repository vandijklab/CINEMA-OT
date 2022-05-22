import scib
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr
from scipy.stats import spearmanr
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

from sklearn.decomposition import FastICA
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.preprocessing import OneHotEncoder
from scipy.stats import ttest_1samp
import harmonypy as hm

def mixscape(adata,obs_label, ref_label, expr_label):
    X_pca1 = adata.obsm['X_pca'][adata.obs[obs_label]==expr_label,:]
    X_pca2 = adata.obsm['X_pca'][adata.obs[obs_label]==ref_label,:]
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X_pca1)
    mixscape_pca = adata.obsm['X_pca']
    mixscapematrix = nbrs.kneighbors_graph(X_pca2).toarray()
    mixscape_pca[adata.obs[obs_label]==ref_label,:] = np.dot(mixscapematrix, mixscape_pca[adata.obs[obs_label]==expr_label,:])/20
    #te2 = adata.X[adata.obs[obs_label]==ref_label,:] - np.matmul(mixscapematrix/np.sum(mixscapematrix,axis=1)[:,None],adata.X[adata.obs[obs_label]==expr_label,:])
    return mixscape_pca, mixscapematrix

def harmony_mixscape(adata,obs_label, ref_label, expr_label):
    meta_data = adata.obs
    data_mat=adata.obsm['X_pca']
    vars_use=[obs_label]
    ho = hm.run_harmony(data_mat, meta_data,vars_use)
    hmdata = ho.Z_corr.T
    X_pca1 = hmdata[adata.obs[obs_label]==expr_label,:]
    X_pca2 = hmdata[adata.obs[obs_label]==ref_label,:]
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X_pca1)
    hmmatrix = nbrs.kneighbors_graph(X_pca2).toarray()
    #te2 = adata.X[adata.obs[obs_label]==ref_label,:] - np.matmul(hmmatrix/np.sum(hmmatrix,axis=1)[:,None],adata.X[adata.obs[obs_label]==expr_label,:])
    return hmdata, hmmatrix


def evaluate_cinema(matrix,gt):
    #includes four statistics: knn-AUC, treatment effect pearson correlation, treatment effect spearman correlation, ttest AUC
    aucdata = np.zeros(gt.shape[0])
    #corr_ = np.zeros(gt.shape[0])
    #scorr_ = np.zeros(gt.shape[0])
    #genesig = np.zeros(gite.shape[1])
    for i in range(gt.shape[0]):
        fpr, tpr, thres = roc_curve(gt[i,:],matrix[i,:])
        aucdata[i] = auc(fpr,tpr)
    #for i in range(ite.shape[0]):
        #corr_[i] = np.corrcoef(ite[i,1000:],gite[i,1000:])[0,1]
        #scorr_[i],pval = spearmanr(ite[i,1000:],gite[i,1000:])
    #    corr_[i] = np.corrcoef(ite[i,:],gite[i,:])[0,1]
    #    scorr_[i],pval = spearmanr(ite[i,:],gite[i,:])        
    return aucdata

def evaluate_batch(sig, adata,obs_label, label, continuity=True,graph_conn=False,pcr=False):
    newsig = sc.AnnData(X=sig, obs = adata.obs)
    sc.pp.pca(newsig,n_comps=min(15,newsig.X.shape[1]-1))
    #newsig.obsm['X_pca'] = newsig.X
    k0=15
    sc.pp.neighbors(newsig, n_neighbors=k0)
    sc.tl.diffmap(newsig, n_comps=min(15,newsig.X.shape[1]-1))
    eigen = newsig.obsm['X_diffmap']
    #newsig_nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(newsig.X)
    #newsig_con = newsig_nbrs.kneighbors_graph(newsig.X)
    #newsig.obsp['connectivities'] = newsig_con
    newsig_metrics = scib.metrics.metrics(adata,newsig,obs_label,obs_label,
        graph_conn_=graph_conn,                            
        pcr_=pcr)
    steps = adata.obs[label].values
    #also we test max correlation to see strong functional dependence between steps and signals, for each state_group population 
    importr("XICOR")
    xicor = ro.r["xicor"]
    if continuity:
        xi = np.zeros(eigen.shape[1])
        pval = np.zeros(eigen.shape[1])
        j = 0
        for source_row in eigen.T:
            rresults = xicor(ro.FloatVector(source_row), ro.FloatVector(steps), pvalue = True)
            xi[j] = np.array(rresults.rx2("xi"))[0]
            pval[j] = np.array(rresults.rx2("pval"))[0]
            j = j+1
        maxcoef = np.max(xi)
        newsig_metrics.rename(index={'trajectory':'trajectory_coef'},inplace=True)
        newsig_metrics.iloc[13,0] = np.max(xi)
    else:
        encoder = OneHotEncoder(sparse=False)
        onehot = encoder.fit_transform(np.array(adata.obs[label].values.tolist()).reshape(-1, 1))
        yi = np.zeros([onehot.shape[1],eigen.shape[1]])
        k = 0
        ind = onehot.T[0] * 0
        m = onehot.T.shape[0]
        for indicator in onehot.T[0:m-1]:
            j = 0
            ind = ind + indicator
            for source_row in eigen.T:
                rresults = xicor(ro.FloatVector(source_row), ro.FloatVector(ind), pvalue = True)
                yi[k,j] = np.array(rresults.rx2("xi"))[0]
                j = j+1
            k = k+1
        
        newsig_metrics.rename(index={'hvg_overlap':'state_coef'},inplace=True)
        newsig_metrics.iloc[12,0] = np.mean(np.max(yi,axis=1))
    
    return newsig_metrics
