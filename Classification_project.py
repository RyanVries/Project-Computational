"""
Created by Ryan de Vries for project computational biology
"""


import pandas as pd
import seaborn as sns
import numpy as np
import datetime
import sklearn
from sklearn.metrics import confusion_matrix

def read_data(file_loc):
    dframe=pd.read_csv(file_loc)
    return dframe

def remove_nan_dframe(dframe,class_sort):
    drop_index=[];
    (r,c)=dframe.shape
    for i in range(0,r):
        column=dframe[class_sort]
        if isinstance(column[i], float):
            drop_index.append(i)
    dframe=dframe.drop(drop_index,axis=0)
    return dframe

def remove_nan_list(lis):
    drop_index=[];
    kept=[];
    r=len(lis)
    for i in range(0,r):
        if isinstance(lis[i], float): 
            drop_index.append(i)
        else:
            kept.append(i)
    for index in sorted(drop_index, reverse=True):
        del lis[index]          
    return lis, kept
    
def make_clustermap(dframe,remove,save_fig,class_sort='lung_carcinoma'):
    if remove==True:
        dframe=remove_nan_dframe(dframe,class_sort)
    cla=dframe[class_sort]
    labels=cla.unique()
    lut = dict(zip(labels, 'rbg'))
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
        extra='_'.join([str(x.day),str(x.hour),str(x.minute),str(x.second)])
        g.savefig('clustermap'+extra+'.png')
    return

def approach_paper(dframe,thresholds):
    """integer 0 implies that the patient is not expected to have a carcinoma and label 1 implies an expected carcinoma"""
    (rows,columns)=dframe.shape
    LC_index=[];
    LC_results=[];
    for pat in range(0,rows):
        for i in range(5,12):
            TM=dframe.columns[i]
            if TM in thresholds.keys() and pat not in LC_index:
                if dframe.iloc[pat,i]>=thresholds[TM]:
                    LC_index.append(pat)
                    LC_results.append(1)
        if pat not in LC_index:
            LC_results.append(0)
    
    return LC_results

def compare_with_ground(dframe,prediction,category):
    """under construction"""
    tru=dframe[category]
    tru=tru.tolist()
    tru, kept=remove_nan_list(tru)   
    ground=[];
    for i in range(0,len(tru)):
        if tru[i]=='Yes' or tru[i]=='NSCLC':       #dit is nu te specifiek
            ground.append(1)
        if tru[i]=='No' or tru[i]=='SCLC':        #dit is nu te specifiek
            ground.append(0)
    pred_down=[];
    for z in range(0,len(prediction)):
        if z in kept:
            pred_down.append(prediction[z])
    con = confusion_matrix(ground,pred_down,labels=[0,1])  #het werkt nu tot hier
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    PPV=tp/(tp+fp)
    NPV=tn/(tn+fn)
    return PPV,NPV,sensitivity,specificity, con
    
file_loc='tumormarkers_lungcancer.csv'
dframe=read_data(file_loc)
make_clustermap(dframe=dframe, remove=True, save_fig=False, class_sort='primary_tumor')

thresholds={'TM_CA15.3 (U/mL)': 35,'TM_CEA (ng/mL)':5,'TM_CYFRA (ng/mL)':3.3,'TM_NSE (ng/mL)':25,'TM_PROGRP (pg/mL)':50,'TM_SCC (ng/mL)':2}
LC_paper=approach_paper(dframe,thresholds)    #hier voorspel ik lung_carcinoma mee
PPV,NPV,sensi,speci,con=compare_with_ground(dframe,LC_paper,'lung_carcinoma')     #moet dit primary tumor zijn of lung carcinoma??
