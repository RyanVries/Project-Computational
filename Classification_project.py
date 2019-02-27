"""
Created by Ryan de Vries for project computational biology
"""

#import all the necessary modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from astropy.table import Table
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn import tree, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
from decimal import getcontext, Decimal  
from sklearn.tree import export_graphviz
import graphviz
from imblearn.over_sampling import SMOTE


def read_data(file_loc):
    '''read the desired data from the csv file as a dataframe'''
    dframe=pd.read_csv(file_loc)
    return dframe

def remove_nan_dframe(dframe,class_sort):
    '''remove patients from the dataframe which contain a Nan value in the specified column and return the new dataframe and the original indexes which were kept '''
    drop_index=[]; #will contain all indexes wchich will have to be removed
    kept=[];  #will contain all kept indexes 
    (r,c)=dframe.shape
    column=dframe[class_sort]   #select column which will have to be evaluated
    for i in range(0,r):  #look at each seperate patient
        if isinstance(column[i], float) or column[i]=='Niet bekend':    #a Nan is classified as a float in python
            drop_index.append(i)   #if it is a Nan the index will have to be removed
        else:
            kept.append(i)    #if not a Nan the index will be kept
    dframe=dframe.drop(drop_index,axis=0)   #drop all Nan indexes
    return dframe, kept

def remove_nan_list(lis):
    '''remove Nan values from the list and also specify which original indexes where kept '''
    drop_index=[];  #will contain all indexes wchich will have to be removed
    kept=[];    #will contain all kept indexes 
    r=len(lis)
    for i in range(0,r):    #look at each seperate patient
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
    cla=dframe[class_sort]   #take the desired column
    labels=cla.unique()    #determine the unique strings in the column
    lut = dict(zip(labels, 'rbgk')) #create dictionary of possible options and assign a color code to each
    row_colors = cla.map(lut)   #provides the corresponding color code for each of the patients and thus indicates the label
    markers=dframe.iloc[:,5:12]  #Tumor markers
    cmap = sns.diverging_palette(250, 10, n=9, as_cmap=True)   #select a color pallete for the clustermap
    g=sns.clustermap(markers, cmap=cmap, metric='euclidean', method='single', col_cluster=False, row_colors=row_colors, z_score=1)   #make clustermap with normalization of the columns

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
    LC_index=[];   #will contain the indexes of patients which are classified as havin LC
    LC_results=[];   #results of the thresholding operation
    for pat in range(0,rows):   #look at each patient 
        for i in range(5,12):   #look at all tumor markers
            TM=dframe.columns[i]
            if TM in thresholds.keys() and pat not in LC_index:     #see if a threshold is present for the tumor marker and if the patient is not already classified as having LC
                if dframe.iloc[pat,i]>=thresholds[TM]:   #if the TM concentration exceeds the threshold at patient to list and classify as having LC
                    LC_index.append(pat)
                    LC_results.append(1)
        if pat not in LC_index:  #if patient does not exceed any of the thresholds classify as not having LC
            LC_results.append(0)
    predictions=LC_results
    
    truth=dframe[category]  
    labels=['No','Yes']
    lut = dict(zip(labels, [0,1]))
    ground = truth.map(lut)   #the ground truth of each patient mapped with the labels dictionary to have a binary problem
    gr=ground.tolist()
    PPV,NPV,sensitivity,specificity,report=evaluate_stats(gr,predictions,labels)  #evaluate the operation by calculaton the programmed statistical values
    print_stats_adv(PPV,NPV,sensitivity,specificity,labels,'Thresholds paper',category_to_investigate)   #provide the statistics in a table
    return PPV,NPV,sensitivity,specificity

def plot_optimal(AUCs,thresholds,TMs,optimal):
    
    for i in range(0,len(AUCs.columns)):
        AUC_list=AUCs[TMs[i]].tolist()
        opt=optimal[TMs[i]]
        label=TMs[i].split(' ')
        plt.figure()  
        plt.plot(thresholds, AUC_list, color='darkorange',label='optimal threshold: %0.2f ' % opt + label[1])
        plt.xlabel('Threshold value '+label[1])
        plt.ylabel('AUC')
        plt.title('Threshold values versus AUC for the tumor marker: '+ label[0])
        plt.plot([opt,opt],[min(AUC_list), max(AUC_list)],linestyle='--',color='black')
        plt.legend(loc="lower right")
        plt.show() 
        
    return 

def optimal_thres(dframe,category='lung_carcinoma'):
    dframe, kept=remove_nan_dframe(dframe,category)
    (rows,columns)=dframe.shape
    TMs=dframe.columns[5:12]
    threshold=np.linspace(0,200,400)
    AUCs=np.zeros((len(threshold),len(TMs)))
    labels=['No','Yes']
    lut = dict(zip(labels, [0,1]))
    y_true=dframe[category].map(lut)
    optimal=dict()
    for mi,marker in enumerate(range(5,12)):
        for index,thres in enumerate(threshold):
            LC_result=np.zeros(rows)
            for pat in range(0,rows):
                if dframe.iloc[pat,marker]>=thres:
                    LC_result[pat]=1
            fpr, tpr, _ = roc_curve(y_true, LC_result)
            AUCs[index,mi]=auc(fpr, tpr)
        place=np.argmax(AUCs[:,mi])
        optimal[TMs[mi]] = threshold[place]
        
    AUCs=pd.DataFrame(AUCs,columns=TMs)
    plot_optimal(AUCs,threshold,TMs,optimal)
    
    return optimal


def optimal_thresCV(dframe,category='lung_carcinoma'):
    dframe, kept=remove_nan_dframe(dframe,category)
    (rows,columns)=dframe.shape
    TMs=dframe.columns[5:12]
    threshold=np.linspace(0,200,400)
    
    labels=['No','Yes']
    lut = dict(zip(labels, [0,1]))
    y_true=dframe[category].map(lut)
    y_true=y_true.tolist()
    y_true=np.array(y_true)
    
    skf = StratifiedKFold(n_splits=10)
    
    overall_optimals=dict()
    for mi,marker in enumerate(range(5,12)):
        AUCs_CV=[]
        optimals=[]
        for train_index, test_index in skf.split(dframe.iloc[:,5:12], y_true):
            AUCs=np.zeros(len(threshold))
            for index,thres in enumerate(threshold):
                LC_result=np.zeros(len(train_index))
                for z,pat in enumerate(train_index):
                    if dframe.iloc[pat,marker]>=thres:
                        LC_result[z]=1
                fpr, tpr, _ = roc_curve(y_true[train_index], LC_result)
                AUCs[index]=auc(fpr, tpr)
            place=np.argmax(AUCs)
            optimal=threshold[place]
            optimals.append(optimal)
        
            predictions=np.zeros(len(test_index))
            for idx,pat in enumerate(test_index):      
                if dframe.iloc[pat,marker]>=optimal:   
                        predictions[idx]=1
            fpr_test, tpr_test, _ = roc_curve(y_true[test_index], predictions)
            AUCs_CV.append(auc(fpr_test, tpr_test))
        label=TMs[mi].split(' ')
        g=plt.figure()
        plt.scatter(optimals,AUCs_CV)
        plt.xlabel('Threshold value '+label[1])
        plt.ylabel('AUC')
        plt.title('Threshold values of cross validated test set versus AUC for the individual tumor marker : '+ label[0])
        plt.show()
            
        spot=np.argmax(AUCs_CV)
        overall_optimals[TMs[mi]]=optimals[spot]
    
    return overall_optimals

def visualize_DT(dtree,feature_names,class_names):
    '''Visualization of the decision tree'''
    export_graphviz(dtree, out_file='tree.dot', feature_names = feature_names,class_names = class_names,rounded = True, proportion = False, precision = 2, filled = True)
    #(graph,) = pydot.graph_from_dot_file('tree.dot')
    #graph=graphviz.Source(dot_data)
    graphviz.render('dot','png','C:/Users/s164616/Documents/MATLAB/Project Computational Biology')
    return 

def prepare_data(dframe,cat,normalize,smote):
    '''prepare the data for the classifier by applying mapping and splitting the data and if specified oversampling and/or normalization'''
    dframe, kept=remove_nan_dframe(dframe,cat)  #remove all Nan since these do not contribute to the classifier
    y_true=dframe[cat]
    labels=y_true.unique()  #determine unique labels
    length=range(0,len(labels))  #provide a integer to each label
    lut = dict(zip(labels, length)) #create dictionary of possible options
    markers=dframe.iloc[:,5:12] #TM
    y_true=y_true.map(lut)   #convert each string to the corresponding integer in the dictionary
        
    if normalize==True and smote!=True:   #scale each of the columns of the tumor markers
        markers = pd.DataFrame(preprocessing.scale(markers.values,axis=0),columns=markers.columns)
        
    X_train, X_test, y_train, y_test = train_test_split(markers.values, y_true, test_size=0.2)   #split the data in a training set and a test set
    
    if smote==True:     #apply synthetic Minority Over-sampling if specified (usually for skewed data distribution)
        sm = SMOTE(random_state=42)   #initialization
        name=markers.columns   #names of the TM's
        X_train,y_train=sm.fit_resample(X_train,y_train)  #apply operation and provide new data
        X_train=pd.DataFrame(X_train,columns=name)   #convert the TM list to a Dataframe
        
    if normalize==True and smote==True:   #scale each of the columns of the tumor markers
        markers = pd.DataFrame(preprocessing.scale(markers.values,axis=0),columns=markers.columns)
        X_train=pd.DataFrame(preprocessing.scale(X_train.values,axis=0),columns=X_train.columns)
        X_test=pd.DataFrame(preprocessing.scale(X_test,axis=0),columns=markers.columns)
    
    return markers, y_true, X_train, X_test, y_train, y_test, labels, lut

def det_CVscore(clf,markers,y_true):
    '''apply cross validation score and determine the mean and standard deviation of the score'''
    score=cross_val_score(clf,markers,y_true,cv=5,scoring='roc_auc')  #cross validation step
    std=np.std(score)
    mn=np.mean(score)
    CV_score={'mean':mn,'std':std}
    return CV_score
    

def decisionT(dframe,cat,save_roc):
    '''Set up a decision tree classifier and train it after which predictions are made for the test set and statistics for this classification are calculated'''
    markers, y_true, X_train, X_test, y_train, y_test, labels, lut=prepare_data(dframe,cat,normalize=False,smote=True) #prepare the data
    clf = tree.DecisionTreeClassifier() #initialization of the classifier
    clf.fit(X_train,y_train)  #fit classifier to training data
    #visualize_DT(clf,dframe.columns[5:12],labels)

    CV_score=det_CVscore(clf,markers.values,y_true)  #apply cross validation and get score
    
    predictions=clf.predict(X_test)       #use reshape(1,-1) on the array when predicting a single array
    PPV,NPV,sensitivity,specificity,report=evaluate_stats(y_test,predictions,labels)  #process the result and provide statistics
    auc_DT=roc_auc(y_test,predictions,cat,save_roc,lut,classifier='Decision Tree classifier')    #AUC and ROC curve of classification
    print_stats_adv(PPV,NPV,sensitivity,specificity,labels,'Decision Tree classifier',cat)  #show statistics in table
    return auc_DT,PPV,NPV,sensitivity,specificity, report, CV_score

def Logistic_clas(dframe,cat,save_roc):
    '''Set up a Logistic Regression classifier and train on data after which the predictions of the test data are evaluated'''
    markers, y_true, X_train, X_test, y_train, y_test, labels, lut=prepare_data(dframe,cat,normalize=True,smote=True)  #prepare data
    
    clf = LogisticRegression(penalty='l2',solver='liblinear')    #initialization of the classifier
    clf.fit(X_train,y_train)  #fitting training set
    
    CV_score=det_CVscore(clf,markers.values,y_true)  #cross validation

    predictions=clf.predict(X_test)       #use reshape(1,-1) on the array when predicting a single array
    PPV,NPV,sensitivity,specificity,report=evaluate_stats(y_test,predictions,labels)  #statistics
    auc_LC=roc_auc(y_test,predictions,cat,save_roc,lut,classifier='Logistic Regression classifier')     #AUC and ROC curve 
    print_stats_adv(PPV,NPV,sensitivity,specificity,labels,'Logistic Regression classifier',cat) #Table of statistics
    return auc_LC,PPV,NPV,sensitivity,specificity, report, CV_score

def evaluate_stats(ground,prediction,labels):
    '''Evaluate the predictions by comparing them with the ground truth and calculate the desired statistical values'''
    cnf_matrix = confusion_matrix(ground,prediction)  #calculate the confusion matrix
    fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  #false positives
    fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)   #false negatives
    tp = np.diag(cnf_matrix)  #true positives
    tn = np.sum(cnf_matrix,axis=(0,1)) - (fp + fn + tp)   #true negatives
    
    report=classification_report(ground,prediction,target_names=labels)   #make a classification report 
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    PPV=tp/(tp+fp)  #positive predictive value
    NPV=tn/(tn+fn)  #negative predictive value
        
    return PPV,NPV,sensitivity,specificity,report

def print_roc(fpr_keras, tpr_keras,auc_keras,save_roc,category,label,classifier):
    '''display the ROC curve and save if specified'''
    g=plt.figure()  
    plt.plot(fpr_keras, tpr_keras, color='darkorange', label='ROC curve (area = %0.2f)' % auc_keras) #plot the ROC curve 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')   #plot a dashed line whch indicates a AUC of 0.5
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if label==False:
        plt.title('Receiver operating characteristic of the ' +classifier+' for the class: '+category)
    else:
        plt.title('Receiver operating characteristic of the ' +classifier+' for the class: '+category+', with the selected label: '+label)
    plt.legend(loc="lower right")
    plt.show()
    if save_roc==True:      #save the figure if wanted with a unique name to prevent overwriting files
        x=datetime.datetime.now()
        extra='_'.join([category,str(x.year),str(x.month),str(x.day),str(x.hour),str(x.minute),str(x.second)])
        g.savefig('ROC_curve'+extra+'.png')
    return
    
def roc_auc(y_true,predictions,category,save_roc,dic,classifier):
    '''calculate the FPR and TPR necessary for the ROC curve and calculate the AUC of this curve'''  
    if len(np.unique(y_true))>2 or len(np.unique(predictions))>2:     #if there are more then two labels then compute ROC curve and ROC area for each class
        fpr = dict() #False positive rates
        tpr = dict()  #True positive rates
        AUC = dict() #AUC values
        labels=np.unique(y_true)  #each of the different integers is seen as a label
        for label in labels:  #define the positive label
                fpr[label], tpr[label], _ = roc_curve(y_true, predictions,pos_label=label)  #roc curves where each label is once seen as the positive
                AUC[label] = auc(fpr[label], tpr[label])  
                for string, n in dic.items():  
                    if n==label:  #find the corresponding class to the integer label
                        label_string=string
                print_roc(fpr[label],tpr[label],AUC[label],save_roc,category,label_string,classifier)  #print all the ROC curves
        return AUC
    else:  #if there are only 2 labels a single ROC curve is enough
        fpr_keras, tpr_keras, _ = roc_curve(y_true, predictions)
        auc_keras = auc(fpr_keras, tpr_keras)
        print_roc(fpr_keras, tpr_keras,auc_keras,save_roc,category,False,classifier) #print ROC curves
        return auc_keras 
    
    
def print_stats(PPV,NPV,sensi,speci,classifier,category):
    'Make a tabel of the relevant statistical values and print this'''
    values='{:.2f} {:.2f} {:.2f} {:.2f}'.format(100*PPV,100*NPV,100*sensi,100*speci)  #string of all statistical values rounded to two decimals
    values=values.split()  # convert string to list
    t=Table([[float(values[0])],[float(values[1])],[float(values[2])],[float(values[3])]],names=('PPV (%)','NPV (%)','Sensitivity (%)','Specificity (%)'),meta={'name':'Statistical values for the '+classifier+' of the class: '+category})  #make Table
    print(t.meta['name'])  #print name of the table
    print(t)  #print Table
    return

def print_stats_adv(PPV,NPV,sensi,speci,labels,classifier,category):
    '''shows for each label the relevant statistical values. where the label is seen as the positive on the stated row'''
    getcontext().prec = 4  #significance of the statistical numbers
    data=[tuple(labels),tuple([Decimal(x) * 100 for x in PPV]),tuple([Decimal(x) * 100 for x in NPV]),tuple([Decimal(x) * 100 for x in sensi]),tuple([Decimal(x) * 100 for x in speci])]  #list of all statistical data
    t=Table(data, names=('positive labels','PPV (%)','NPV (%)','Sensitivity (%)','Specificity (%)'),meta={'name':'Statistical values for the '+classifier+' of the class: '+category})  #make the table
    print(t.meta['name'])  #print name of the table
    print(t)  #print Table
    return
    
category_to_investigate='lung_carcinoma'
file_loc='tumormarkers_lungcancer.csv'
dframe=read_data(file_loc)    #read data
make_clustermap(dframe=dframe, remove=True, save_fig=False, class_sort=category_to_investigate)

thresholds={'TM_CA15.3 (U/mL)': 35,'TM_CEA (ng/mL)':5,'TM_CYFRA (ng/mL)':3.3,'TM_NSE (ng/mL)':25,'TM_PROGRP (pg/mL)':50,'TM_SCC (ng/mL)':2}
if category_to_investigate=='lung_carcinoma':
    PPV,NPV,sensi,speci=approach_paper(dframe,thresholds,category_to_investigate)        
aucDT,PPV,NPV,sensitivity,specificity, report, CV_score=decisionT(dframe,category_to_investigate,save_roc=False)
aucLC,PPV2,NPV2,sensitivity2,specificity2, report2, CV_score2=Logistic_clas(dframe,category_to_investigate,save_roc=False)
