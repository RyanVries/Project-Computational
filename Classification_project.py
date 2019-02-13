"""
Created by Ryan de Vries for project computational biology
"""


import pandas as pd
import seaborn as sns
import datetime
import sklearn

def read_data(file_loc):
    dframe=pd.read_csv(file_loc)
    return dframe

def remove_nan_dframe(dframe):
    drop_index=[];
    (r,c)=dframe.shape
    for i in range(0,r):
        if isinstance(dframe.iloc[i,1], float) and isinstance(dframe.iloc[i,2], float):
            drop_index.append(i)
    dframe=dframe.drop(drop_index,axis=0)
    return dframe
    
def make_cluster(dframe,remove,save_fig,class_sort='lung_carcinoma'):
    if remove==True:
        dframe=remove_nan_dframe(dframe)
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
        extra='_'.join([str(x.hour),str(x.minute)])
        g.savefig('clustermap'+extra+'.png')
    return

def approach_paper(dframe,thresholds):
    (rows,columns)=dframe.shape
    LC_index=[];
    for pat in range(0,rows):
        for i in range(5,12):
            TM=dframe.columns[i]
            if TM in thresholds.keys() and pat not in LC_index:
                if dframe.iloc[pat,i]>=thresholds[TM]:
                    LC_index.append(pat)
    return LC_index

def compare_with_ground(dframe,prediction):
    """under construction"""
    clinical=dframe['primary_tumor']
    "clframe,kept=remove_nan(clinical)"
    con=sklearn.metrics.confusion_matrix(clframe,prediction[kept])
    tn, fp, fn, tp = con([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    
    return
    
file_loc='tumormarkers_lungcancer.csv'
dframe=read_data(file_loc)
make_cluster(dframe=dframe, remove=True, save_fig=False, class_sort='lung_carcinoma')

thresholds={'TM_CA15.3 (U/mL)': 35,'TM_CEA (ng/mL)':5,'TM_CYFRA (ng/mL)':3.3,'TM_NSE (ng/mL)':25,'TM_PROGRP (pg/mL)':50,'TM_SCC (ng/mL)':2}
LC_paper=approach_paper(dframe,thresholds)
compare_with_ground(dframe,LC_paper)
