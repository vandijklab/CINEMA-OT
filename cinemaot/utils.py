import gseapy as gp
import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
import scanpy as sc
#import scib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import kstest

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()


def wgcna_module_scores(de_adata):
    """
    Caculate gene modules and soft connectivity scores using WGCNA
    Remark: There are some scaling for mean gene expression and the variance. Here we select all highly variable genes remained after pre-processing. 
    """
    wgcna = importr('WGCNA')
    #variance = np.var(de_matrix, axis=0)
    #genes_to_select = np.argsort(-variance) < n_variable_genes
    #de_trimmed = de_matrix[:,genes_to_select]
    modules = wgcna.blockwiseModules(de_adata.X, numericLabels=True)
    # calculate top hub genes per module
    soft_connectivities = wgcna.softConnectivity(de_adata.X)
    return pd.DataFrame({
        'gene_name': de_adata.var_names,
        'module': modules.rx2('colors').astype(int),
        'soft_connectivity': soft_connectivities
    })


def dominantcluster(adata,ctobs,clobs):
    clustername = []
    clustertime = np.zeros(adata.obs[ctobs].value_counts().values.shape[0])
    for i in adata.obs[clobs].value_counts().sort_index().index.values:
        tmp = adata.obs[ctobs][adata.obs[clobs]==i].value_counts().sort_index()
        ind = np.argmax(tmp.values)
        clustername.append(tmp.index.values[ind] + str(int(clustertime[ind])))
        clustertime[ind] = clustertime[ind] + 1
    return clustername

def assignleiden(adata,ctobs,clobs,label):
    clustername = dominantcluster(adata,ctobs,clobs)
    ss = adata.obs[clobs].values.tolist()
    for i in range(len(ss)):
        ss[i] = clustername[int(ss[i])]
    adata.obs[label] = ss
    return

def clustertest_synergy(adata1,adata2,clobs,thres,fthres,path):
    # In this simplified function, we return the gene set only. The function is only designed for synergy computation.
    mkup = []
    mkdown = []
    for i in list(set(adata1.obs[clobs].values.tolist())):
        adata = adata1
        clusterindex = (adata.obs[clobs].values==i)
        tmpte = adata.X[clusterindex,:]
        clustername = i
        pv = np.zeros(tmpte.shape[1])
        for k in range(tmpte.shape[1]):
            st, pv[k] = wilcoxon(tmpte[:,k],zero_method='zsplit')
        genenames = adata.var_names.values
        upindex = (((pv<thres)*1) * ((np.median(tmpte,axis=0)>0)*1) * (np.abs(np.median(tmpte,axis=0))>fthres))>0
        downindex = (((pv<thres)*1) * ((np.median(tmpte,axis=0)<0)*1)* (np.abs(np.median(tmpte,axis=0))>fthres))>0
        allindex = (((pv<thres)*1) * (np.abs(np.median(tmpte,axis=0))>fthres))>0
        upgenes1 = genenames[upindex]
        downgenes1 = genenames[downindex]
        allgenes1 = genenames[allindex]
        adata = adata2
        clusterindex = (adata.obs[clobs].values==i)
        tmpte = adata.X[clusterindex,:]
        clustername = i
        pv = np.zeros(tmpte.shape[1])
        for k in range(tmpte.shape[1]):
            st, pv[k] = wilcoxon(tmpte[:,k],zero_method='zsplit')
        genenames = adata.var_names.values
        upindex = (((pv<thres)*1) * ((np.median(tmpte,axis=0)>0)*1) * (np.abs(np.median(tmpte,axis=0))>fthres))>0
        downindex = (((pv<thres)*1) * ((np.median(tmpte,axis=0)<0)*1)* (np.abs(np.median(tmpte,axis=0))>fthres))>0
        allindex = (((pv<thres)*1) * (np.abs(np.median(tmpte,axis=0))>fthres))>0
        upgenes2 = genenames[upindex]
        downgenes2 = genenames[downindex]
        allgenes2 = genenames[allindex]
        up1syn = list(set(upgenes1.tolist()) - set(upgenes2.tolist()))
        up2syn = list(set(upgenes2.tolist()) - set(upgenes1.tolist()))
        down1syn = list(set(downgenes1.tolist()) - set(downgenes2.tolist()))
        down2syn = list(set(downgenes2.tolist()) - set(downgenes1.tolist()))
        allgenes = list(set(up1syn) | set(up2syn) | set(down1syn) | set(down2syn))
        enr_up1 = gp.enrichr(gene_list=up1syn, gene_sets=['./genelist/h.all.v7.5.1.symbols.gmt','./genelist/c5.go.bp.v7.5.1.symbols.gmt'],
                     no_plot=True,organism='Human',
                     outdir='./genelist/prerank_report_kegg', format='png')
        enr_up2 = gp.enrichr(gene_list=up2syn, gene_sets=['./genelist/h.all.v7.5.1.symbols.gmt','./genelist/c5.go.bp.v7.5.1.symbols.gmt'],
                     no_plot=True,organism='Human',
                     outdir='./genelist/prerank_report_kegg', format='png')
        enr_down1 = gp.enrichr(gene_list=down1syn, gene_sets=['./genelist/h.all.v7.5.1.symbols.gmt','./genelist/c5.go.bp.v7.5.1.symbols.gmt'],
                     no_plot=True,organism='Human',
                     outdir='./genelist/prerank_report_kegg', format='png')
        enr_down2 = gp.enrichr(gene_list=down2syn, gene_sets=['./genelist/h.all.v7.5.1.symbols.gmt','./genelist/c5.go.bp.v7.5.1.symbols.gmt'],
                     no_plot=True,organism='Human',
                     outdir='./genelist/prerank_report_kegg', format='png')
        if not enr_up1.results.empty:
            enr_up1.results.iloc[enr_up1.results['Adjusted P-value'].values<1e-3,:].to_csv(path+'/Up1'+clustername+'.csv')
        if not enr_up2.results.empty:
            enr_up2.results.iloc[enr_up2.results['Adjusted P-value'].values<1e-3,:].to_csv(path+'/Up2'+clustername+'.csv')
        if not enr_down1.results.empty:
            enr_down1.results.iloc[enr_down1.results['Adjusted P-value'].values<1e-3,:].to_csv(path+'/Down1'+clustername+'.csv')
        if not enr_down2.results.empty:
            enr_down2.results.iloc[enr_down2.results['Adjusted P-value'].values<1e-3,:].to_csv(path+'/Down2'+clustername+'.csv')
        upgenes1df = pd.DataFrame(index=up1syn)
        upgenes2df = pd.DataFrame(index=up2syn)
        downgenes1df = pd.DataFrame(index=down1syn)
        downgenes2df = pd.DataFrame(index=down2syn)
        allgenesdf = pd.DataFrame(index=allgenes)
        upgenes1df.to_csv(path+'/Upnames1'+clustername+'.csv')
        upgenes2df.to_csv(path+'/Upnames2'+clustername+'.csv')
        downgenes1df.to_csv(path+'/Downnames1'+clustername+'.csv')
        downgenes2df.to_csv(path+'/Downnames2'+clustername+'.csv')
        allgenesdf.to_csv(path+'/names'+clustername+'.csv')

    return


def clustertest(adata,clobs,thres,fthres,label,path):
    # Changed from ttest to Wilcoxon test
    clusternum = int(np.max((np.asfarray(adata.obs[clobs].values))))
    genenum = np.zeros([clusternum+1])
    mk = []
    for i in range(clusternum+1):
        clusterindex = (np.asfarray(adata.obs[clobs].values)==i)
        tmpte = adata.X[clusterindex,:]
        clustername = adata.obs[label][clusterindex][0]
        pv = np.zeros(tmpte.shape[1])
        for k in range(tmpte.shape[1]):
            st, pv[k] = wilcoxon(tmpte[:,k],zero_method='zsplit')
        genenames = adata.var_names.values
        upindex = (((pv<thres)*1) * ((np.median(tmpte,axis=0)>0)*1) * (np.abs(np.median(tmpte,axis=0))>fthres))>0
        downindex = (((pv<thres)*1) * ((np.median(tmpte,axis=0)<0)*1)* (np.abs(np.median(tmpte,axis=0))>fthres))>0
        allindex = (((pv<thres)*1) * (np.abs(np.median(tmpte,axis=0))>fthres))>0
        upgenes = genenames[upindex]
        downgenes = genenames[downindex]
        allgenes = genenames[allindex]
        mk.extend(allgenes.tolist())
        mk = list(set(mk))
        genenum[i] = np.sum(((pv<thres)*1) * ((np.abs(np.mean(tmpte,axis=0))>fthres)))
        enr_up = gp.enrichr(gene_list=upgenes.tolist(), gene_sets=['./genelist/h.all.v7.5.1.symbols.gmt','./genelist/c5.go.bp.v7.5.1.symbols.gmt'],
                     no_plot=True,organism='Human',
                     outdir='./genelist/prerank_report_kegg', format='png')
        enr_down = gp.enrichr(gene_list=downgenes.tolist(), gene_sets=['./genelist/h.all.v7.5.1.symbols.gmt','./genelist/c5.go.bp.v7.5.1.symbols.gmt'],
                     no_plot=True,organism='Human',
                     outdir='./genelist/prerank_report_kegg', format='png')
        enr = gp.enrichr(gene_list=allgenes.tolist(), gene_sets=['./genelist/h.all.v7.5.1.symbols.gmt','./genelist/c5.go.bp.v7.5.1.symbols.gmt'],
                     no_plot=True,organism='Human',
                     outdir='./genelist/prerank_report_kegg', format='png')
        if not enr_up.results.empty:
            enr_up.results.iloc[enr_up.results['Adjusted P-value'].values<1e-3,:].to_csv(path+'/Up'+clustername+'.csv')
        if not enr_down.results.empty:
            enr_down.results.iloc[enr_down.results['Adjusted P-value'].values<1e-3,:].to_csv(path+'/Down'+clustername+'.csv')
        if not enr.results.empty:
            enr.results.iloc[enr.results['Adjusted P-value'].values<1e-3,:].to_csv(path+'/'+clustername+'.csv')
        upgenesdf = pd.DataFrame(index=upgenes)
        downgenesdf = pd.DataFrame(index=downgenes)
        allgenesdf = pd.DataFrame(index=allgenes)
        upgenesdf.to_csv(path+'/Upnames'+clustername+'.csv')
        downgenesdf.to_csv(path+'/Downnames'+clustername+'.csv')
        allgenesdf.to_csv(path+'/names'+clustername+'.csv')
        if not enr.results.empty:
            if i == 0:
                df = enr.results.transpose().iloc[4:5,:]
                df.columns = enr.results['Term'][:]
                df.index.values[0] = clustername
            else:
                tmp = enr.results.transpose().iloc[4:5,:]
                tmp.columns = enr.results['Term'][:]
                tmp.index.values[0] = clustername
                df = pd.concat([df,tmp])
    #df.values = -np.log10(df.values)
    #DF = sc.AnnData(df.transpose())
    #sc.pl.clustermap(DF,cmap='viridis', col_cluster=False)
    return genenum, df, mk


def concordance_map(confounder,response,obs_label, cl_label, condition):
    #deprecatedc
    cf = confounder[confounder.obs[obs_label] == condition,:]
    cf.obs['res_cl'] = response.obs[cl_label].values
    aswmatrix = np.zeros([len(list(set(cf.obs['res_cl'].values.tolist()))),len(list(set(cf.obs['res_cl'].values.tolist())))])
    indnummatrix = pd.DataFrame(None,list(set(cf.obs['res_cl'].values.tolist())),list(set(cf.obs['res_cl'].values.tolist())))
    k = 0
    #return aswmatrix
    for i in list(set(cf.obs['res_cl'].values.tolist())):
        l = 0
        for j in list(set(cf.obs['res_cl'].values.tolist())):
            if i != j:
                tmpcf = cf[cf.obs['res_cl'].isin([i,j]),:].copy()
                sc.pp.pca(tmpcf)
                encoder = OneHotEncoder(sparse=False)
                onehot = encoder.fit_transform(np.array(tmpcf.obs['res_cl'].values.tolist()).reshape(-1, 1))
                label = onehot[:,0]
                lc = LogisticRegression(penalty='l1',solver='liblinear',C=1)
                lc.fit(tmpcf.X, label)
                prob = lc.predict_proba(tmpcf.X)
                prob1 = prob[label==1,0]
                prob2 = prob[label==0,0]
                st, pv = kstest(prob1,prob2)
                #yi = np.zeros([onehot.shape[1],eigen.shape[1]])
                aswmatrix[k,l] = -np.log10(pv+1e-20)
                if np.sum(lc.coef_!=0)>0:
                    indnummatrix.iloc[k,l] = str(np.argwhere(lc.coef_[0] !=0)[:,0].tolist())[1:-1]
            else:
                aswmatrix[k,l] = 0
            l = l + 1
        k = k + 1
    aswmatrix = pd.DataFrame(aswmatrix,list(set(cf.obs['res_cl'].values.tolist())),list(set(cf.obs['res_cl'].values.tolist())))
    return aswmatrix, indnummatrix


def coarse_matching(de,de_label,ref,ref_label,ot,scaling=1e6):
    coarse_ot = pd.DataFrame(index=sorted(set(de.obs[de_label].values.tolist())),columns=sorted(set(ref.obs[ref_label].values.tolist())),dtype=float)
    for i in set(de.obs[de_label].values.tolist()):
        for j in set(ref.obs[ref_label].values.tolist()):
            tmp_ot = ot[de.obs[de_label]==i,:]
            coarse_ot[j][i] = np.mean(tmp_ot[:,ref.obs[ref_label]==j]) * scaling
    return coarse_ot


