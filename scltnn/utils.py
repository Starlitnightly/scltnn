r"""
Miscellaneous utilities
(In development )
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from distfit import distfit
from scipy.stats import norm,dweibull
from scipy import stats
from keras.models import load_model
import pandas as pd
import numpy as np
from keras.layers import Dense,Conv1D,Convolution1D,Activation,MaxPooling1D,Flatten,Dropout

from keras.callbacks import TensorBoard
from keras import regularizers
import scipy.sparse
import sklearn.preprocessing
import sklearn.utils.extmath
import scanpy as sc
import anndata

from anndata import AnnData

from typing import Optional, Union
Array = Union[np.ndarray, scipy.sparse.spmatrix]

def cal_high_variable_genes(adata,n_top_genes=10000,save=True):
    r"""calualate the high variable genes by seurat_v3 of scanpy

    Arguments
    ---------
    adata:
        Annotated data matrix.
    n_top_genes:
        the num of the high variable genes selected
    
    Returns
    -------
    high_variable_gene:
        the list of high variable genes

    """
    print('......calculate high_variable_genes',n_top_genes)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    if save==True:
        adata=adata[:,adata.var['highly_variable']==True]
    else:
        adata=adata

    high_variable_gene=adata.var[adata.var['highly_variable']==True].tolist()
    return high_variable_gene

def tfidf(X: Array) -> Array:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def lsi(
        adata: anndata.AnnData, n_components: int = 100,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi

def cal_lsi(adata,n_components=100,n_iter=15):
    r"""calculate the latent semantic index of scRNA-seq

    Arguments
    ---------
    adata:
        Annotated data matrix.
    n_components:
        the num latent dimension of LSI 
    n_iter:
        the num of iter
    
    """
    print('......calculate lsi')
    lsi(adata, n_components=n_components, n_iter=n_iter)
    
def cal_paga(adata,
            resolution=1.0,
            n_neighbors=10,
            n_pcs=40,
            ):
    r"""calculate the paga graph by community detection of leiden

    Arguments
    ---------
    adata:
        Annotated data matrix.
    resolution:
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn’t accept a `resolution_parameter`.
    n_neighbors:
        The size of local neighborhood (in terms of number of neighboring data points) 
        used for manifold approximation. Larger values result in more global views of 
        the manifold, while smaller values result in more local data being preserved. 
        In general values should be in the range 2 to 100. If knn is True, number of 
        nearest neighbors to be searched. If knn is False, a Gaussian kernel width is 
        set to the distance of the n_neighbors neighbor.
    n_pcs:
        Use this many PCs. If n_pcs==0 use .X if use_rep is None.
    
    """
    print('......calculate paga')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=resolution)
    sc.tl.paga(adata, groups='leiden')

def cal_model_time(adata,model_path):
    r"""predict the latent time by primriary ANN model

    Arguments
    ---------
    adata:
        Annotated data matrix.
    model_path:
        the path of the ANN model constructed by primriart training
        
    """
    try:
        adata.obsm['X_lsi']
    except NameError:
        print('Error, you need to calculate the lsi first')
        return 
    else:
        print('lsi check success')
    
    print('......predict model_time')
    model=load_model(model_path)
    X_val=adata.obsm['X_lsi']
    PredValSet=(model.predict(X_val))

    print('.......adata add model time (p_time)')
    adata.obs['p_time']=PredValSet.T[0]
    print('.......adata add model rev time (p_time_r)')
    adata.obs['p_time_r']=1-PredValSet.T[0]
    

def find_high_correlation_gene(adata,rev=False):

    r"""Calculate the Pearson Correlation between gene and LTNN_time
    Arguments
    ---------
    adata
        the anndata performed LTNN analysis
    rev
        the selection of LTNN_time or LTNN_time_r
    Returns
    -------
    LTNN_time_Pearson
        the pandas of LTNN_time_Pearson 
    adata
        the anndata calculated by find_high_correlation_gene
    """

    """
    # Extract data from count matrix
    """
    if rev==True:
        pd1=adata.obs.loc[:,['LTNN_time_r']]
    else:
        pd1 = adata.obs.loc[:,['LTNN_time']]
    pd2 = pd.DataFrame(adata.X.toarray(),columns = adata.var_names,index = adata.obs_names )

    """
    # Calculate the Pearson Correlation
    """
    LTNN_time_Cor = np.arange(len(adata.var.index),dtype=float)  
    for i in range(len(pd2.columns)):
        res = stats.pearsonr(pd1.to_numpy().flatten(),pd2.iloc[:,i].to_numpy())
        LTNN_time_Cor[i] = float(res[0])

    """
    # Assign Pearson_Correlation to adata
    """
    LTNN_time_Pearson = pd.DataFrame(LTNN_time_Cor,index=pd2.columns)
    adata.var.loc[:,'Pearson_correlation'] = LTNN_time_Pearson.iloc[:,0].to_list()
    """
    # Extract the Pearson Correlation
    """
    LTNN_time_Pearson['feautre'] = LTNN_time_Pearson.index
    LTNN_time_Pearson.columns = ['correlation','feature']
    LTNN_time_Pearson['abs_correlation'] = LTNN_time_Pearson['correlation'].abs()
    LTNN_time_Pearson['sig']='+'
    LTNN_time_Pearson.loc[(LTNN_time_Pearson.correlation<0),'sig'] = '-'
    LTNN_time_Pearson=LTNN_time_Pearson.sort_values('correlation',ascending=False)
    return LTNN_time_Pearson,adata