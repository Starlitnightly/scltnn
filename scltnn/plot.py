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


sc_color=['#7CBB5F', '#368650', '#A499CC', '#5E4D9A','#5E4D9A','#78C2ED',
                                  '#866017','#9F987F','#E0DFED','#EF7B77','#279AD7','#F0EEF0','#1F577B',
                                  '#A56BA7','#E0A7C8','#E069A6','#941456','#FCBC10','#EAEFC5','#01A0A7',
                                  '#75C8CC','#F0D7BC','#D5B26C','#D5DA48','#B6B812','#9DC3C3','#A89C92',
                                  '#FEE00C','#FEF2A1']

def plot_high_correlation_heatmap(
    adata,LTNN_time_Pearson,
    meta=None,
    number=10,
    rev=True,
    fontsize=10,
    meta_legend=False,
    meta_legend_kws={'ncol':2,'interval':7,'distance':23},
    **kwarg
):
    r"""
    Heatmap of gene changes with LTNN_time

    Arguments
    ---------
    adata
        the anndata performed LTNN analysis and find_high_correlation_gene
    LTNN_time_Pearson
        the calculation result of LTNN_time_Person 
    number
        the num of genes to visualization
    rev
        the selection of LTNN_time or LTNN_time_r
    fontsize
        the fontsize of whole figure
    meta_legend
        the visualization clusters of cell in 'adata.obs'
    meta_legend
        the kwags of meta_legend 
        -ncol: the ncol of legend: meta_legend_kws['ncol']=2 
        -distance: the distance of legend and heatmap: meta_legend_kws['distance']=23
        -interval: the interval of multi-legend meta_legend_kws['interval']=7
    **kwarg
        the kwarg of seaborn.clustermap

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
    df.index =  df[LTNN_ticks]
    df.drop([LTNN_ticks],axis=1,inplace=True)
    df.index=adata.obs.sort_values(LTNN_ticks).index
    
    #return df
    if meta!=None:
        meta_colors={}
        adata_index=adata.obs.sort_values(LTNN_ticks).index
        color_map=pd.DataFrame(index=adata_index)
        for i in meta:
            meta_color=dict(zip(adata.obs.loc[adata_index,i].value_counts().index,sc_color))
            meta_colors[i]=meta_color
            color_map[i]=adata.obs.loc[adata_index,i].map(meta_color)
        

    """
    # Visualization
    """
    #pp=plt.figure(figsize=figsize)
    #ax=pp.add_subplot(1,1,1)
    if meta!=None:
        a=sns.clustermap(df.T, cbar_kws={'shrink':0.5,'ticks':[0, 0.50, 1],},
                         robust=True, xticklabels=False,
                           mask=False,yticklabels=True,
                            col_cluster=False,col_colors=color_map,**kwarg
                         )
    else:
        a=sns.clustermap(df.T, cbar_kws={'shrink':0.5,'ticks':[0, 0.50, 1], },
                         robust=True, xticklabels=False,
                           mask=False,yticklabels=True,
                            col_cluster=False,**kwarg
                         )
    #x0, _y0, _w, _h = a.cbar_pos
    #a.ax_cbar.set_position([x0, 0.9, a.ax_row_dendrogram.get_position().width, 0.02])
    #a.ax_cbar.set_title('colorbar title')
    #a.ax_cbar.tick_params(axis='x', length=10,size=fontsize)
    #for spine in a.ax_cbar.spines:
    #    a.ax_cbar.spines[spine].set_color('crimson')
    #    a.ax_cbar.spines[spine].set_linewidth(2)
    #a.ax_cbar.xaxis.label.set_size(fontsize)
        #labels=a.ax_col_colors.yaxis.get_ticklabels()
        #plt.setp(labels, fontsize=fontsize)
    
    #set the tick fontsize
    a.ax_heatmap.yaxis.set_tick_params(labelsize=fontsize)
    a.ax_heatmap.xaxis.set_tick_params(labelsize=fontsize)
    
    if meta_legend==True:
        '''
        meta_legend_kws['ncol']=2
        meta_legend_kws['distance']=23
        meta_legend_kws['interval']=7
        '''
        k=0
        for i in meta:
            yy=[]
            for label in adata.obs.loc[adata_index,i].unique():
                b1=a.ax_col_dendrogram.bar(0, 0, color=meta_colors[i][label],
                                        label=label, linewidth=0)
                yy.append(b1)

            legend3 = plt.legend(yy, adata.obs.loc[adata_index,i].unique(), fontsize=fontsize,
                                  loc='center',title=i, ncol=meta_legend_kws['ncol'], bbox_to_anchor=(meta_legend_kws['distance']+k, -2, 0.5, 0.5), )
            plt.gca().add_artist(legend3)  
            k+=meta_legend_kws['interval']
        
        labels=a.ax_col_colors.yaxis.get_ticklabels()
        plt.setp(labels, fontsize=fontsize)
            
    
    
    return a


def plot_origin_tesmination(adata,origin,tesmination):
    r"""
    plot the origin and tesmination cell of scRNA-seq
    
    Arguments
    ---------
    adata
        the anndata performed LTNN analysis
    origin
        the origin cell list/numpy.nparray
    tesmination
        the tesmination cell list/numpy.nparray

    Returns
    -------
    ax
        the axex subplot of heatmap
    """
    start_mao=[]
    start_mao_name=[]
    for i in adata.obs['leiden']:
        if i in origin:
            start_mao.append('-1')
            start_mao_name.append('Origin')
        elif i in tesmination:
            start_mao.append('1')
            start_mao_name.append('Tesmination')
        else:
            start_mao.append('0')
            start_mao_name.append('Other')
    adata.obs['mao']=start_mao
    adata.obs['mao']=adata.obs['mao'].astype('category')
    adata.obs['mao_name']=start_mao_name
    adata.obs['mao_name']=adata.obs['mao_name'].astype('category')
    nw=adata.obs['mao_name'].cat.categories
    mao_color={
        'Origin':'#e25d5d',
        'Other':'white',
        'Tesmination':'#a51616'
    }
    adata.uns['mao_name_colors'] = nw.map(mao_color)
    #return adata
    
    ax=sc.pl.embedding(adata,basis='umap',edges=True,edges_color='#f4897b',
                   color='mao_name',title='Origin and Tesmination',alpha=0.6,
                   frameon=False,legend_fontsize=13,show=False)
    #t.plot([0,10],[0,10])

    circle1_loc=adata[adata.obs['mao']=='-1'].obsm['X_umap'].mean(axis=0)
    circle1_max=adata[adata.obs['mao']=='-1'].obsm['X_umap'].max(axis=0)
    circle1_r=circle1_loc[0]-circle1_max[0]
    circle1 = plt.Circle(circle1_loc, circle1_r*1.2, color='#e25d5d',fill=False,ls='--',lw=2)

    circle2_loc=adata[adata.obs['mao']=='1'].obsm['X_umap'].mean(axis=0)
    circle2_max=adata[adata.obs['mao']=='1'].obsm['X_umap'].max(axis=0)
    circle2_r=circle2_loc[0]-circle2_max[0]
    circle2 = plt.Circle(circle2_loc, circle2_r*1.2, color='#a51616',fill=False,ls='--',lw=2)

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    return ax
