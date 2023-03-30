import scib
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# In this newer version we use the Python implementation of xicor
# import rpy2.robjects as ro
# import rpy2.robjects.numpy2ri
# import rpy2.robjects.pandas2ri
# from rpy2.robjects.packages import importr
# rpy2.robjects.numpy2ri.activate()
# rpy2.robjects.pandas2ri.activate()

from scipy.stats.stats import pearsonr
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import pairwise_distances
from . import sinkhorn_knopp as skp

from sklearn.preprocessing import OneHotEncoder
from scipy.stats import ttest_1samp
import harmonypy as hm

def mixscape(adata,obs_label, ref_label, expr_label, nn=20, return_te = True):
    X_pca1 = adata.obsm['X_pca'][adata.obs[obs_label]==expr_label,:]
    X_pca2 = adata.obsm['X_pca'][adata.obs[obs_label]==ref_label,:]
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(X_pca1)
    mixscape_pca = adata.obsm['X_pca'].copy()
    mixscapematrix = nbrs.kneighbors_graph(X_pca2).toarray()
    mixscape_pca[adata.obs[obs_label]==ref_label,:] = np.dot(mixscapematrix, mixscape_pca[adata.obs[obs_label]==expr_label,:])/20
    if return_te:
        te2 = adata.X[adata.obs[obs_label]==ref_label,:] - (mixscapematrix/np.sum(mixscapematrix,axis=1)[:,None]) @ (adata.X[adata.obs[obs_label]==expr_label,:])
        return mixscape_pca, mixscapematrix, te2
    else:
        return mixscape_pca, mixscapematrix

def harmony_mixscape(adata,obs_label, ref_label, expr_label,nn=20, return_te = True):
    meta_data = adata.obs
    data_mat=adata.obsm['X_pca']
    vars_use=[obs_label]
    ho = hm.run_harmony(data_mat, meta_data,vars_use)
    hmdata = ho.Z_corr.T
    X_pca1 = hmdata[adata.obs[obs_label]==expr_label,:]
    X_pca2 = hmdata[adata.obs[obs_label]==ref_label,:]
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(X_pca1)
    hmmatrix = nbrs.kneighbors_graph(X_pca2).toarray()
    if return_te:
        te2 = adata.X[adata.obs[obs_label]==ref_label,:] - np.matmul(hmmatrix/np.sum(hmmatrix,axis=1)[:,None],adata.X[adata.obs[obs_label]==expr_label,:])
        return hmdata, hmmatrix, te2
    else:
        return hmdata, hmmatrix

def OT(adata,obs_label, ref_label, expr_label,thres=0.01, return_te = True):
    cf1 = adata.obsm['X_pca'][adata.obs[obs_label]==expr_label,0:20]
    cf2 = adata.obsm['X_pca'][adata.obs[obs_label]==ref_label,0:20]
    r = np.zeros([cf1.shape[0],1])
    c = np.zeros([cf2.shape[0],1])
    r[:,0] = 1/cf1.shape[0]
    c[:,0] = 1/cf2.shape[0]
    sk = skp.SinkhornKnopp(setr=r,setc=c,epsilon=1e-2)
    dis = pairwise_distances(cf1,cf2)
    e = thres * adata.obsm['X_pca'].shape[1]
    af = np.exp(-dis * dis / e)
    ot = sk.fit(af).T
    OT_pca = adata.obsm['X_pca'].copy()
    OT_pca[adata.obs[obs_label]==ref_label,:] = np.matmul(ot/np.sum(ot,axis=1)[:,None],OT_pca[adata.obs[obs_label]==expr_label,:])
    if return_te:
        te2 = adata.X[adata.obs[obs_label]==ref_label,:] - np.matmul(ot/np.sum(ot,axis=1)[:,None],adata.X[adata.obs[obs_label]==expr_label,:])
        return OT_pca, ot, te2
    else:
        return OT_pca, ot


def evaluate_cinema(matrix,ite,gt,gite):
    #includes four statistics: knn-AUC, treatment effect pearson correlation, treatment effect spearman correlation, ttest AUC
    aucdata = np.zeros(gt.shape[0])
    corr_ = np.zeros(gt.shape[0])
    scorr_ = np.zeros(gt.shape[0])
    #genesig = np.zeros(gite.shape[1])
    for i in range(gt.shape[0]):
        fpr, tpr, thres = roc_curve(gt[i,:],matrix[i,:])
        aucdata[i] = auc(fpr,tpr)
    for i in range(ite.shape[0]):
        corr_[i], pval = pearsonr(ite[i,1000:],gite[i,1000:])
        scorr_[i],pval = spearmanr(ite[i,1000:],gite[i,1000:])
        corr_[i], pval = pearsonr(ite[i,:],gite[i,:])
        scorr_[i],pval = spearmanr(ite[i,:],gite[i,:])        
    return np.median(aucdata), np.median(corr_), np.median(scorr_)

def evaluate_batch(sig, adata,obs_label, label, continuity,asw=True,silhouette=True,graph_conn=True,pcr=True,nmi=True,ari=True,diff_coefs=False):
    #Label is a list!!!
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
    newsig_metrics = scib.metrics.metrics(adata,newsig,obs_label,label[0],
        isolated_labels_asw_= asw,
        graph_conn_= graph_conn,
        silhouette_ = silhouette,
        nmi_=nmi,
        ari_=ari,                            
        pcr_=pcr)
    if diff_coefs:
        for i in range(len(label)):
            steps = adata.obs[label[i]].values
            #also we test max correlation to see strong functional dependence between steps and signals, for each state_group population 
            if continuity[i]:
                xi = np.zeros(eigen.shape[1])
                #pval = np.zeros(eigen.shape[1])
                j = 0
                for source_row in eigen.T:
                    #rresults = xicor(ro.FloatVector(source_row), ro.FloatVector(steps), pvalue = True)
                    xi_obj = Xi(source_row,steps.astype(np.float))
                    xi[j] = xi_obj.correlation
                    j = j+1
                maxcoef = np.max(xi)
                #newsig_metrics.rename(index={'trajectory':'trajectory_coef'},inplace=True)
                #newsig_metrics.iloc[13,0] = np.max(xi)
                newsig_metrics.loc[label[i]] = maxcoef
            else:
                encoder = OneHotEncoder(sparse=False)
                onehot = encoder.fit_transform(np.array(adata.obs[label[i]].values.tolist()).reshape(-1, 1))
                yi = np.zeros([onehot.shape[1],eigen.shape[1]])
                k = 0
                #ind = onehot.T[0] * 0
                m = onehot.T.shape[0]
                for indicator in onehot.T[0:m-1]:
                    j = 0
                    #ind = ind + indicator
                    for source_row in eigen.T:
                        xi_obj = Xi(source_row,indicator*1)
                        yi[k,j] = xi_obj.correlation
                        j = j+1
                    k = k+1
        
            #newsig_metrics.rename(index={'hvg_overlap':'state_coef'},inplace=True)
            #newsig_metrics.iloc[12,0] = np.mean(np.max(yi,axis=1))
                newsig_metrics.loc[label[i]] = np.mean(np.max(yi,axis=1))
        
    return newsig_metrics


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