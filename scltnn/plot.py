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


def calculate_pearson_correlation(
    adata: AnnData,
):

  """
   Calculate the pearson correlation between Gene and LTNN_time
  """
  """
  # 从计数矩阵中提取数据
  """
  pd1 = adata.obs.loc[:,['LTNN_time']]
  pd2 = pd.DataFrame(adata.X.toarray(),columns = adata.var_names,index = adata.obs_names )
 
  """
  # 计算pearson相关性系数
  """
  LTNN_time_Cor = np.arange(len(adata.var.index),dtype=float)  
  for i in range(len(pd2.columns)):
    res = stats.pearsonr(pd1.to_numpy().flatten(),pd2.iloc[:,i].to_numpy())
    LTNN_time_Cor[i] = float(res[0])
  
   """
  # 将Pearson_Correlation赋值给adata
   """
  LTNN_time_Pearson = pd.DataFrame(LTNN_time_Cor,index=pd2.columns)
  adata.var.loc[:,'Pearson_correlation'] = LTNN_time_Pearson.iloc[:,0].to_list()
  """
  # 提取出和LTNN_time相关的相关性大小
   """
  LTNN_time_Pearson['feautre'] = LTNN_time_Pearson.index
  LTNN_time_Pearson.columns = ['correlation','feature']
  LTNN_time_Pearson['abs_correlation'] = LTNN_time_Pearson['correlation'].abs()
  LTNN_time_Pearson['sig']='+'
  LTNN_time_Pearson.loc[(LTNN_time_Pearson.correlation<0),'sig'] = '-'
  return LTNN_time_Pearson,adata

def heatmap(
    adata: AnnData,
    number: int=10,
    LTNN_time_Pearson=None,
    cmap = None,
    
):
  """
  Heatmap of gene changes with LTNN_time
  """
  """
  # 提取出和LTNN_Time相关性最大的Gene
  """
  LTNN_time_Pearson=LTNN_time_Pearson.sort_values('correlation',ascending=False)
  LTNN_time_Pearson_pos=LTNN_time_Pearson.sort_values('correlation',ascending=True).iloc[:number]
  LTNN_time_Pearson_neg=LTNN_time_Pearson.iloc[:number].sort_values('correlation',ascending=False)
  LTNN_time_Pearson_p=pd.concat([LTNN_time_Pearson_pos,LTNN_time_Pearson_neg], axis=0, join='outer')
  LTNN_time_Pearson_p

  """
  # 设置number参数方便后面的排序
  """
  adata.obs['number'] = np.arange(len(adata.obs.index))
  new_obs_index=[i for i in adata.obs.sort_values('LTNN_time').index]
  adata[new_obs_index]

  """
  # 从anndata结构中提取出关键参数
  """
  markers = LTNN_time_Pearson_p.index
  df = sc.get.obs_df(adata, keys=list(markers)+['LTNN_time'])
  df.sort_values('LTNN_time',inplace=True)
  df.index =  df.LTNN_time
  df.drop(['LTNN_time'],axis=1,inplace=True)

  """
  # 可视化
  """
  pp=plt.figure(figsize=(12,10))
  ax=pp.add_subplot(1,1,1)
  ax = sns.heatmap(df.T, cmap=cmap, cbar=True, robust=True, xticklabels=False )
