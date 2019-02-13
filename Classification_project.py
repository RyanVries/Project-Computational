"""
Created by Ryan de Vries for project computational biology
"""


import pandas as pd
import seaborn as sns
import time
import datetime

def read_data(file_loc):
    dframe=pd.read_csv(file_loc)
    return dframe

def remove_nan(dframe):
    drop_index=[];
    (r,c)=dframe.shape
    for i in range(0,r):
        if isinstance(dframe.iloc[i,1], float) and isinstance(dframe.iloc[i,2], float):
            drop_index.append(i)
    dframe=dframe.drop(drop_index,axis=0)
    return dframe
    
def make_cluster(dframe,remove,save_fig,class_sort='lung_carcinoma'):
    if remove==True:
        dframe=remove_nan(dframe)
        
    cla=dframe[class_sort]
    lut = dict(zip(cla.unique(), 'rbg'))
    labels=cla.unique()
    row_colors = cla.map(lut)
    markers=dframe.iloc[:,[5,6,7,8,9,10,11]]
    g=sns.clustermap(markers, metric='correlation', method='single', col_cluster=False, row_colors=row_colors, z_score=0)

    for label in labels:
        g.ax_col_dendrogram.bar(0, 0, color=lut[label],
                            label=label, linewidth=0)
    g.ax_col_dendrogram.legend(loc='center', ncol=3)
    g.ax_heatmap.set_title('Clustermap of the protein biomarkers with labels of the different known '+str(class_sort)+' classes')
    if save_fig==True:
        x=datetime.datetime.now()
        ex='_'.join([str(x.hour),str(x.minute)])
        g.savefig('clustermap'+ex+'.png')
    return

file_loc='tumormarkers_lungcancer.csv'
dframe=read_data(file_loc)
make_cluster(dframe=dframe, remove=True, save_fig=False, class_sort='lung_carcinoma')


