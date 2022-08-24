r"""
Plotting functions
"""

import matplotlib.pyplot as plt 

import matplotlib.axes as ma
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import sklearn.metrics
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

def set_publication_params() -> None:
    r"""
    Set publication-level figure parameters
    """
    sc.set_figure_params(
        scanpy=True, dpi_save=600, vector_friendly=True, format="pdf",
        facecolor=(1.0, 1.0, 1.0, 0.0), transparent=False
    )
    rcParams["savefig.bbox"] = "tight"

def get_publication_colors(color_name) -> dict:
    r"""
    Get the color set of ltnn 

    Arguments
    ---------
    color_name
        the latent color of publication
        - ltnn: the raw color
        - age: the age color
        - red: the red color
        - green: the green color
        - blue: the blue color
        - yellow: the yellow color
        - purple: the purple color
    
    Returns
    -------
    c_u
        color dict construct by three part:
        colors, cmap, and cmap_r
    """

    if color_name=='ltnn':
        colors=['#F7828A',"#F9C7C6","#FDFAF3","#D4E3D0","#9CCCA4",]
    elif color_name=='age':
        colors=['#008c5f','#3b9868','#b0df83','#ed7c6a','#e54746','#177cb0','#a3dfdf']
    elif color_name=='red':
        colors=['#e88989','#e25d5d','#a51616']
    elif color_name=='green':
        colors=['#75c99f','#3b9868','#0d6a3b']
    elif color_name=='blue':
        colors=['#4f61c7','#177cb0','#3f5062']
    elif color_name=='yellow':
        colors=['#fbc16f','#f2973a','#f6aa00']
    elif color_name=='purple':
        colors=['#c57bac','#c453a4','#9b1983']



    c = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
    colors.reverse()
    c_r=LinearSegmentedColormap.from_list('Custom', colors, len(colors))
    colors.reverse()
    c_u={
        'colors':colors,
        'cmap':c,
        'cmap_r':c_r
    }
    return c_u



def plot_high_correlation_heatmap(
    adata,
    LTNN_time_Pearson,
    number=10,
    cmap = None,
    rev=True,

):
    r"""Heatmap of gene changes with LTNN_time

    Arguments
    ---------
    adata
        the anndata performed LTNN analysis and find_high_correlation_gene
    number
        the num of genes to visualization
    LTNN_time_Pearson
        the set of LTNN_time_Person
    cmap
        the colormap of heatmap
    rev
        the selection of LTNN_time or LTNN_time_r

    Returns
    -------
    ax
        the axex subplot of heatmap

    """

    """
    # Extract the the maximal Pearson Correlation
    """
    if rev==True:
        LTNN_ticks='LTNN_time_r'
    else:
        LTNN_ticks='LTNN_time'


    LTNN_time_Pearson_pos=LTNN_time_Pearson.sort_values('correlation',ascending=True).iloc[:number]
    LTNN_time_Pearson_neg=LTNN_time_Pearson.iloc[:number].sort_values('correlation',ascending=False)
    LTNN_time_Pearson_p=pd.concat([LTNN_time_Pearson_pos,LTNN_time_Pearson_neg], axis=0, join='outer')
    LTNN_time_Pearson_p

    """
    # Sort
    """
    adata.obs['number'] = np.arange(len(adata.obs.index))
    new_obs_index=[i for i in adata.obs.sort_values(LTNN_ticks).index]
    adata[new_obs_index]

    """
    # Extract data from anndata
    """
    markers = LTNN_time_Pearson_p.index
    df = sc.get.obs_df(adata, keys=list(markers)+[LTNN_ticks])
    df.sort_values(LTNN_ticks,inplace=True)
    df.index =  df.LTNN_time
    df.drop([LTNN_ticks],axis=1,inplace=True)

    """
    # Visualization
    """
    pp=plt.figure(figsize=(12,10))
    ax=pp.add_subplot(1,1,1)
    ax = sns.heatmap(df.T, cmap=cmap, cbar=True, robust=True, xticklabels=False )
    return ax
