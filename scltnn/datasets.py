import time
import requests
import os
from keras.models import load_model
import scanpy as sc

def model_downloader(url,path,title):
    r"""model downloader
    
    Arguments
    ---------
    url
        the download url of model
    path
        the save path of model
    title
        the name of model
    
    """
    if os.path.isfile(path):
        print("......Loading dataset from {}".format(path))
        return path
    else:
        print("......Downloading dataset save to {}".format(path))
        
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))
    
    
    start = time.time()
    size = 0
    res = requests.get(url, stream=True)

    chunk_size = 102400
    content_size = int(res.headers["content-length"]) 
    if res.status_code == 200:
        print('......[%s Size of file]: %0.2f MB' % (title, content_size/chunk_size/10.24))
        with open(path, 'wb') as f:
            for data in res.iter_content(chunk_size=chunk_size):
                f.write(data)
                size += len(data) 
                print('\r'+ '......[Downloader]: %s%.2f%%' % ('>'*int(size*50/content_size), float(size/content_size*100)), end='')
        end = time.time()
        print('\n' + ".......Finish！%s.2f s" % (end - start))
    
    return path


def Cancer_CD8():
    r""" Download the dataset

    Arguments
    ---------
    datasets_name:
        'ZhangZemin_CD8+':the dataset of BCL of CD8+ T cells,GEO:GSE156728
    """
    datasets_name='ZhangZemin_CD8+'
    _datasets={
        'ZhangZemin_CD8+':'https://figshare.com/ndownloader/files/36870252',
    }
    model_path = model_downloader(url=_datasets[datasets_name],path='datasets/{}.h5ad'.format(datasets_name),title=datasets_name)
    adata = sc.read_h5ad(model_path)
    return adata


def Zebrafish():
    r""" Download the dataset

    Arguments
    ---------
    datasets_name:
        'Zebrafish':the dataset of zebrafish cells
    """
    datasets_name='Zebrafish'
    _datasets={
        'Zebrafish':'https://figshare.com/ndownloader/files/27265280',
    }
    model_path = model_downloader(url=_datasets[datasets_name],path='datasets/{}.h5ad'.format(datasets_name),title=datasets_name)
    adata = sc.read_h5ad(model_path)
    return adata

def Pancreas():
    r""" Download the dataset

    Arguments
    ---------
    datasets_name:
        'Pancreas':the dataset of Pancreas cells
    """
    datasets_name='Pancreas'
    _datasets={
        'Pancreas':'https://figshare.com/ndownloader/files/36892242',
    }
    model_path = model_downloader(url=_datasets[datasets_name],path='datasets/{}.h5ad'.format(datasets_name),title=datasets_name)
    adata = sc.read_h5ad(model_path)
    return adata    