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
from keras.models import load_model
import pandas as pd
import numpy as np
from keras.layers import Dense,Conv1D,Convolution1D,Activation,MaxPooling1D,Flatten,Dropout

from keras.callbacks import TensorBoard
from keras import regularizers
import scglue
import scanpy as sc

from anndata import AnnData

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
    scglue.data.lsi(adata, n_components=n_components, n_iter=n_iter)
    
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
    

