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
from sklearn.model_selection import train_test_split
import numpy as np
from decimal import getcontext, Decimal

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
    markers=dframe.iloc[:,5:12]
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

def approach_paper(dframe,thresholds,category):
    """use the specified thresholds from the paper to classify each patient (LC=1 and no LC=0)"""
    dframe, kept=remove_nan_dframe(dframe,category)
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
    predictions=LC_results
    
    truth=dframe[category]
    labels=['Yes','No']
    lut = dict(zip(labels, [1,0]))
    ground = truth.map(lut)
    gr=ground.tolist()
    PPV,NPV,sensitivity,specificity=evaluate_stats(gr,predictions)
    return PPV,NPV,sensitivity,specificity


def decisionT(dframe,cat,save_roc):
    '''Set up a decision tree classifier and train this with 75% of the data and evaluate afterwards with the 25% of test data by showing the ROC curve and its AUC'''
    dframe, kept=remove_nan_dframe(dframe,cat)
    labels=dframe[cat].unique()
    length=range(0,len(labels))
    lut = dict(zip(labels, length)) #create dictionary of possible options
    markers=dframe.iloc[:,5:12]
    y_true=dframe[cat]
    X_train, X_test, y_train, y_test = train_test_split(markers, y_true, test_size=0.2)
    
    train_res_mapped = y_train.map(lut)
    train_mark=X_train
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_mark.values,train_res_mapped)
    
    test_res_map=y_test.map(lut)
    test_mark=X_test
    predictions=clf.predict(test_mark.values)       #use reshape(1,-1) on the array when predicting a single array
    PPV,NPV,sensitivity,specificity=evaluate_stats(test_res_map,predictions)
    auc_DT=roc_auc(test_res_map,predictions,cat,save_roc,lut)    #moet nog verandert worden voor multiclass van tumor_subtype
    
    print_stats_adv(PPV,NPV,sensitivity,specificity,labels)
    return auc_DT,PPV,NPV,sensitivity,specificity

def evaluate_stats(ground,prediction):
    '''Evaluate the predictions by comparing them with the ground truth and calculate the desired statistical values'''
    cnf_matrix = confusion_matrix(ground,prediction)
    if len(np.unique(ground))>2 or len(np.unique(prediction))>2:   
        fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
        fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        tp = np.diag(cnf_matrix)
        tn = np.sum(cnf_matrix,axis=(0,1)) - (fp + fn + tp) 
    else:
        fp = cnf_matrix[0,1]
        fn = cnf_matrix[1,0]
        tp = cnf_matrix[1,1]
        tn = cnf_matrix[0,0]
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    PPV=tp/(tp+fp)
    NPV=tn/(tn+fn)
        
    return PPV,NPV,sensitivity,specificity

def print_roc(fpr_keras, tpr_keras,auc_keras,save_roc,category,label):
    '''display the ROC curve and save if specified'''
    g=plt.figure()
    plt.plot(fpr_keras, tpr_keras, color='darkorange', label='ROC curve (area = %0.2f)' % auc_keras)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if label==False:
        plt.title('Receiver operating characteristic of the class: '+category)
    else:
        plt.title('Receiver operating characteristic of the class: '+category+', with the selected label: '+label)
    plt.legend(loc="lower right")
    plt.show()
    if save_roc==True:      #save the figure if wanted with a unique name to prevent overwriting files
        x=datetime.datetime.now()
        extra='_'.join([category,str(x.year),str(x.month),str(x.day),str(x.hour),str(x.minute),str(x.second)])
        g.savefig('ROC_curve'+extra+'.png')
    return
    
def roc_auc(y_true,predictions,category,save_roc,dic):
    '''calculate the FPR and TPR necessary for the ROC curve and calculate the AUC of this curve'''  
    if len(np.unique(y_true))>2 or len(np.unique(predictions))>2:   #######nog mee nezig want werkt niet voor tumor subtype
        fpr = dict()
        tpr = dict()
        AUC = dict()
        labels=np.unique(y_true)
        for label in labels:
                fpr[label], tpr[label], _ = roc_curve(y_true, predictions,pos_label=label)
                AUC[label] = auc(fpr[label], tpr[label])
                for s, n in dic.items():
                    if n==label:
                        label_string=s
                print_roc(fpr[label],tpr[label],AUC[label],save_roc,category,label_string)
        return AUC
    else:
        fpr_keras, tpr_keras, _ = roc_curve(y_true, predictions)
        auc_keras = auc(fpr_keras, tpr_keras)
        print_roc(fpr_keras, tpr_keras,auc_keras,save_roc,category,label=False) 
        return auc_keras 
    
    
def print_stats(PPV,NPV,sensi,speci):
    'Make a tabel of the relevant statistical values and print this'''
    values='{:.2f} {:.2f} {:.2f} {:.2f}'.format(100*PPV,100*NPV,100*sensi,100*speci)
    values=values.split()
    t=Table([[float(values[0])],[float(values[1])],[float(values[2])],[float(values[3])]],names=('PPV (%)','NPV (%)','Sensitivity (%)','Specificity (%)'),meta={'name':'Statistical values'})
    print(t)
    return

def print_stats_adv(PPV,NPV,sensi,speci,labels):
    getcontext().prec = 4
    data=[tuple(labels),tuple([Decimal(x) * 100 for x in PPV]),tuple([Decimal(x) * 100 for x in NPV]),tuple([Decimal(x) * 100 for x in sensi]),tuple([Decimal(x) * 100 for x in speci])]
    t=Table(data, names=('labels','PPV (%)','NPV (%)','Sensitivity (%)','Specificity (%)'),meta={'name':'Statistical values'})
    print(t)
    return
    
category_to_investigate='tumor_subtype'
file_loc='tumormarkers_lungcancer.csv'
dframe=read_data(file_loc)
make_clustermap(dframe=dframe, remove=True, save_fig=False, class_sort=category_to_investigate)

thresholds={'TM_CA15.3 (U/mL)': 35,'TM_CEA (ng/mL)':5,'TM_CYFRA (ng/mL)':3.3,'TM_NSE (ng/mL)':25,'TM_PROGRP (pg/mL)':50,'TM_SCC (ng/mL)':2}
if category_to_investigate=='lung_carcinoma' or category_to_investigate=='primary_tumor':
    PPV,NPV,sensi,speci=approach_paper(dframe,thresholds,category_to_investigate)        
    print_stats(PPV,NPV,sensi,speci)
aucDT,PPV,NPV,sensitivity,specificity=decisionT(dframe,category_to_investigate,save_roc=False)