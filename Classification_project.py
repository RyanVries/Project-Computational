"""
Created by Ryan de Vries for project computational biology
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from astropy.table import Table
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import tree
import graphviz
import numpy as np

def read_data(file_loc):
    '''read the desired data from the csv file as a dataframe'''
    dframe=pd.read_csv(file_loc)
    return dframe

def remove_nan_dframe(dframe,class_sort):
    '''remove patients from the dataframe which contain a Nan value in the specified column name and return the new dataframe and the original indexes which were kept '''
    drop_index=[];
    kept=[];
    (r,c)=dframe.shape
    column=dframe[class_sort]
    for i in range(0,r):
        if isinstance(column[i], float) or column[i]=='Niet bekend':    #a Nan is classified as a float in python
            drop_index.append(i)
        else:
            kept.append(i)
    dframe=dframe.drop(drop_index,axis=0)
    return dframe, kept

def remove_nan_list(lis):
    '''remove Nan values from the list and also specify which original indexes where kept '''
    drop_index=[];
    kept=[];
    r=len(lis)
    for i in range(0,r):
        if isinstance(lis[i], float): 
            drop_index.append(i)
        else:
            kept.append(i)
    for index in sorted(drop_index, reverse=True):  #elements need to be deleted from back to front to prevent indexing issues
        del lis[index]          
    return lis, kept
    
def make_clustermap(dframe,remove,save_fig,class_sort='lung_carcinoma'):
    '''make a clustermap of the selected column in the dataframe together with the corresponding labels of each patient'''
    if remove==True:    #remove Nan's if specified
        dframe, kept=remove_nan_dframe(dframe,class_sort)
    cla=dframe[class_sort]
    labels=cla.unique()
    lut = dict(zip(labels, 'rbgk')) #create dictionary of possible options
    row_colors = cla.map(lut)
    markers=dframe.iloc[:,[5,6,7,8,9,10,11]]
    g=sns.clustermap(markers, metric='correlation', method='single', col_cluster=False, row_colors=row_colors, z_score=0)

    for label in labels:     #add the labels of each patient next to the clustermap
        g.ax_col_dendrogram.bar(0, 0, color=lut[label],
                            label=label, linewidth=0)
    g.ax_col_dendrogram.legend(loc='center', ncol=3)
    g.ax_heatmap.set_title('Clustermap of the protein biomarkers with labels of the different known '+str(class_sort)+' classes')
    if save_fig==True:      #save the figure if wanted with a unique name to prevent overwriting files
        x=datetime.datetime.now()
        extra='_'.join([str(x.year),str(x.month),str(x.day),str(x.hour),str(x.minute),str(x.second)])
        g.savefig('clustermap'+extra+'.png')
    return

def approach_paper(dframe,thresholds):
    """use the specified thresholds from the paper to classify each patient (LC=1 and no LC=0)"""
    (rows,columns)=dframe.shape
    LC_index=[];
    LC_results=[];
    for pat in range(0,rows):
        for i in range(5,12):
            TM=dframe.columns[i]
            if TM in thresholds.keys() and pat not in LC_index:     #see if a threshold is present for the tumor marker and if the patient is not already classified as having LC
                if dframe.iloc[pat,i]>=thresholds[TM]:
                    LC_index.append(pat)
                    LC_results.append(1)
        if pat not in LC_index:
            LC_results.append(0)
    
    return LC_results


def decisionT(dframe,cat,save_roc):
    dframe, kept=remove_nan_dframe(dframe,cat)
    labels=dframe[cat].unique()
    length=range(0,len(labels))
    lut = dict(zip(labels, length)) #create dictionary of possible options
    msk = np.random.rand(len(dframe)) < 0.75
    df_train=dframe[msk]
    y_train=df_train[cat]
    train_res_mapped = y_train.map(lut)
    markers=dframe.iloc[:,[5,6,7,8,9,10,11]]
    train_mark=markers[msk]
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_mark.values,train_res_mapped)
    
    df_test=dframe[~msk]
    y_test=df_test[cat]
    test_res_map=y_test.map(lut)
    test_mark=markers[~msk]
    predictions=clf.predict(test_mark.values)       #use reshape(1,-1) on the array when predicting a single array
    auc_DT=roc_auc(test_res_map,predictions,cat,save_roc)
    return auc_DT

def compare_with_ground_binary(dframe,prediction,category):
    '''Evaluate the predictions by comparing them with the ground truth and calculate the desired statistical values'''
    frame, kept=remove_nan_dframe(dframe,category)
    tru=frame[category]
    pred_down=[];
    for z in range(0,len(prediction)):   #only keep the prediction if the true value is known
        if z in kept:
            pred_down.append(prediction[z])
    labels=['Yes','No']
    lut = dict(zip(labels, [1,0]))
    ground = tru.map(lut)
    cnf_matrix = confusion_matrix(ground,pred_down,labels=[0,1])  
    fp = cnf_matrix[0,1]
    fn = cnf_matrix[1,0]
    tp = cnf_matrix[1,1]
    tn = cnf_matrix[0,0]
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    PPV=tp/(tp+fp)
    NPV=tn/(tn+fn)
        
    return PPV,NPV,sensitivity,specificity, cnf_matrix

def print_roc(fpr_keras, tpr_keras,auc_keras,save_roc,category):
    '''display the ROC curve and save if specified'''
    g=plt.figure()
    plt.plot(fpr_keras, tpr_keras, color='darkorange', label='ROC curve (area = %0.2f)' % auc_keras)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of the class: '+category)
    plt.legend(loc="lower right")
    plt.show()
    if save_roc==True:      #save the figure if wanted with a unique name to prevent overwriting files
        x=datetime.datetime.now()
        extra='_'.join([str(x.year),str(x.month),str(x.day),str(x.hour),str(x.minute),str(x.second)])
        g.savefig('ROC_curve'+extra+'.png')
    return
    
def roc_auc(y_true,predictions,category,save_roc):
    '''calculate the FPR and TPR necessary for the ROC curve and calculate the AUC of this curve''' 
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, predictions)
    auc_keras = auc(fpr_keras, tpr_keras)
    print_roc(fpr_keras, tpr_keras,auc_keras,save_roc,category)
    return auc_keras 
    
    
def print_stats(PPV,NPV,sensi,speci):
    'Make a tabel of the relevant statistical values and print this'''
    values='{:.2f} {:.2f} {:.2f} {:.2f}'.format(100*PPV,100*NPV,100*sensi,100*speci)
    values=values.split()
    t=Table([[float(values[0])],[float(values[1])],[float(values[2])],[float(values[3])]],names=('PPV (%)','NPV (%)','Sensitivity (%)','Specificity (%)'),meta={'name':'Statistical values'})
    print(t)
    return
    
category_to_investigate='lung_carcinoma'
file_loc='tumormarkers_lungcancer.csv'
dframe=read_data(file_loc)
make_clustermap(dframe=dframe, remove=True, save_fig=False, class_sort=category_to_investigate)

thresholds={'TM_CA15.3 (U/mL)': 35,'TM_CEA (ng/mL)':5,'TM_CYFRA (ng/mL)':3.3,'TM_NSE (ng/mL)':25,'TM_PROGRP (pg/mL)':50,'TM_SCC (ng/mL)':2}
LC_paper=approach_paper(dframe,thresholds)    
PPV,NPV,sensi,speci,cnf=compare_with_ground_binary(dframe,LC_paper,category_to_investigate)     
print_stats(PPV,NPV,sensi,speci)
aucDT=decisionT(dframe,category_to_investigate,save_roc=False)