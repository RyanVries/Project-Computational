"""
Created by Ryan de Vries for project computational biology
"""

#import all the necessary modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from astropy.table import Table
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, f1_score, precision_score
from sklearn import tree, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
from decimal import getcontext, Decimal  
from sklearn.tree import export_graphviz
import graphviz
from imblearn.over_sampling import SMOTE
from random import randint
import warnings
warnings.filterwarnings("ignore")


def read_data(file_loc):
    '''read the desired data from the csv file as a dataframe'''
    dframe=pd.read_csv(file_loc)
    return dframe

def remove_nan_dframe(dframe,class_sort):
    '''remove patients from the dataframe which contain a Nan value in the specified column and return the new dataframe and the original indexes which were kept '''
    drop_index=[]; #will contain all indexes which will have to be removed
    kept=[];  #will contain all kept indexes 
    for i in dframe.index:  #look at each seperate patient
        if isinstance(dframe.loc[i,class_sort], float) or dframe.loc[i,class_sort]=='Niet bekend':    #a Nan is classified as a float in python
            drop_index.append(i)   #if it is a Nan the index will have to be removed
        else:
            kept.append(i)    #if not a Nan the index will be kept
    dframe=dframe.drop(drop_index,axis=0)   #drop all Nan indexes
    return dframe, kept

def remove_nan_markers(dframe):
    '''remove the patients with unknown concentrations of the tumor markers'''
    drop_index=[]; #will contain all indexes wchich will have to be removed  
    TMs=dframe.columns[6:13]  #names of columns with the tumor markers
    for marker in TMs:   #look at each column which contains a TM
        for pat in dframe.index:  #look at each patientin the dataframe
            if np.isnan(dframe.loc[pat,marker])==True and pat not in drop_index:   #if the patient has a Nan as concentraton add to list
                drop_index.append(pat)
    dframe=dframe.drop(drop_index,axis=0)  #drop all patient with unknown TM(s)
    return dframe

def remove_nan_int(dframe,cat='age'):
    '''remove patients from the dataframe which contain a Nan value in the specified column with integers/floats and return the new dataframe'''
    drop_index=[]; #will contain all indexes which will have to be removed
    kept=[];  #will contain all kept indexes 
    for i in dframe.index:  #look at each seperate patient
        if np.isnan(dframe.loc[i,cat])==True and i not in drop_index:  #if the value is a Nan then add to list  
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
    markers=dframe.iloc[:,6:13]  #Tumor markers
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
    truth=dframe[category]  
    labels=['No','Yes']
    lut = dict(zip(labels, [0,1]))
    
    ground = truth.map(lut)   #the ground truth of each patient mapped with the labels dictionary to have a binary problem
    gr=ground.tolist()
    #statistics for each individual marker
    PPVm=np.zeros(7)  
    NPVm=np.zeros(7)
    sensm=np.zeros(7)
    specm=np.zeros(7)
    
    LC_results=np.zeros(rows)   #results of the thresholding operation
    for i in range(6,13):   #look at all tumor markers
        TM=dframe.columns[i]  #current marker
        LC_marker=np.zeros(rows)   #classification for each individual marker
        if TM in thresholds.keys():     #see if a threshold is present for the tumor marker
            for pat in range(0,rows):   #look at each patient 
                if dframe.iloc[pat,i]>=thresholds[TM]:   #if the TM concentration exceeds the threshold at patient to list and classify as having LC
                    LC_results[pat]=1
                    LC_marker[pat]=1
            P,N,S,E,_=evaluate_stats(gr,LC_marker,labels)  #calculate the statistics for each individual marker
            PPVm[i-6]=P[1]
            NPVm[i-6]=N[1]
            sensm[i-6]=S[1]
            specm[i-6]=E[1]
    print_stats_adv(PPVm,NPVm,sensm,specm,dframe.columns[6:13],'Individual thresholds',category_to_investigate)   #provide the statistics in a table for each individual marker
    predictions=LC_results
    
    PPV,NPV,sensitivity,specificity,report=evaluate_stats(gr,predictions,labels)  #evaluate the operation by calculaton the programmed statistical values
    print_stats_adv(PPV,NPV,sensitivity,specificity,labels,'Thresholds paper',category_to_investigate)   #provide the statistics in a table
    return PPV,NPV,sensitivity,specificity,report

def plot_optimal(AUCs,thresholds,TMs,optimal):
    '''plot the continuous AUC values against the corresponding thresholds for each marker'''
    for i in range(0,len(AUCs.columns)):  #loop over each marker
        AUC_list=AUCs[TMs[i]].tolist()   #take the right marker
        opt=optimal[TMs[i]]    #take the optimal threshold value
        label=TMs[i].split(' ')
        plt.figure()  
        plt.plot(thresholds, AUC_list, color='darkorange',label='optimal threshold: %0.2f ' % opt + label[1])   #plot the continuous AUCs
        plt.xlabel('Threshold value '+label[1])
        plt.ylabel('AUC')
        plt.title('Threshold values versus AUC for the tumor marker: '+ label[0])
        plt.plot([opt,opt],[min(AUC_list), max(AUC_list)],linestyle='--',color='black')  #plot the optimal threshold value as a dashed line
        plt.legend(loc="lower right")
        plt.show() 
        
    return 

def optimal_thres(dframe,category='lung_carcinoma'):
    '''determine the optimal thresholds for each marker by optimalization of the AUC'''
    dframe, kept=remove_nan_dframe(dframe,category)  #remove Nans
    (rows,columns)=dframe.shape
    TMs=dframe.columns[6:13]    #names of all tumor markers
    threshold=np.linspace(0,200,400)   #define possible thresholds
    AUCs=np.zeros((len(threshold),len(TMs)))   #make room in memory for the AUCs
    labels=['No','Yes']
    lut = dict(zip(labels, [0,1]))
    y_true=dframe[category].map(lut)    #map the true classification to binary values
    optimal=dict()   #dictionary to store the best threshold values
    for mi,marker in enumerate(range(6,13)):  #look at each marker separately
        for index,thres in enumerate(threshold):   #loop over all of the possible threshold values
            LC_result=np.zeros(rows)  #make room in memory for classification
            for pat in range(0,rows):   #look at each patient
                if dframe.iloc[pat,marker]>=thres:   #classification process
                    LC_result[pat]=1
            fpr, tpr, _ = roc_curve(y_true, LC_result)   #determine roc of each threshold
            AUCs[index,mi]=auc(fpr, tpr)   #determine AUC of each threshold
        place=np.argmax(AUCs[:,mi])   #determine index of best AUC
        optimal[TMs[mi]] = threshold[place]   #add optimal threshold to dictionary with the corresponding marker
        
    AUCs=pd.DataFrame(AUCs,columns=TMs)  #convert to dataframe
    plot_optimal(AUCs,threshold,TMs,optimal)   #plot the AUC values and optimal threshold
    
    return optimal

def optimal_thresCV(dframe,category='lung_carcinoma'):
    '''determine the optimal threshold for each marker by applying cross validation and optimalization of the AUC'''
    dframe, kept=remove_nan_dframe(dframe,category)  #remove Nans
    (rows,columns)=dframe.shape
    TMs=dframe.columns[6:13]   #names of tumor markers
    threshold=np.linspace(0,200,400)   #define threshold range
    
    labels=['No','Yes']
    lut = dict(zip(labels, [0,1]))
    y_true=dframe[category].map(lut)
    y_true=y_true.tolist()
    y_true=np.array(y_true)  #numpy array of the ground truth
    
    skf = StratifiedKFold(n_splits=10)   #initialization of the cross validation
    
    overall_optimals=dict()  #dictionary which will contain the best threshold for each marker
    for mi,marker in enumerate(range(6,13)):   #look at each marker
        AUCs_CV=[]   #will contain the AUCs of a marker
        optimals=[]   #optimal thresholds for each CV set
        for train_index, test_index in skf.split(dframe, y_true):  #apply cross validation
            AUCs=np.zeros(len(threshold))   #will contain the AUCs for all thresholds of the training set
            for index,thres in enumerate(threshold):  #loop over all possible thresholds
                LC_result=np.zeros(len(train_index))   #will contain classification for this threshold
                for z,pat in enumerate(train_index):   #loop over patients in training set
                    if dframe.iloc[pat,marker]>=thres:   #classify
                        LC_result[z]=1  
                fpr, tpr, _ = roc_curve(y_true[train_index], LC_result)   #roc for each threshold
                AUCs[index]=auc(fpr, tpr)  #add AUC to list for this training set
            place=np.argmax(AUCs)   #place best AUC for this training set
            optimal=threshold[place]  #optimal threshold for this CV training set
            optimals.append(optimal)  #extend the optimal thresholds for each CV set
        
            predictions=np.zeros(len(test_index))  #make space in memory for this CV set
            for idx,pat in enumerate(test_index):   #look at each patient in the test set   
                if dframe.iloc[pat,marker]>=optimal:   #classify with the optimal threshold determined for the training set
                        predictions[idx]=1
            fpr_test, tpr_test, _ = roc_curve(y_true[test_index], predictions)   #roc of this CV test set
            AUCs_CV.append(auc(fpr_test, tpr_test))  #AUC of this CV test set
        label=TMs[mi].split(' ')
        plt.figure()
        plt.scatter(optimals,AUCs_CV)
        plt.xlabel('Threshold value '+label[1])
        plt.ylabel('AUC')
        plt.title('Threshold values of cross validated test set versus AUC for the individual tumor marker : '+ label[0])
        plt.show()
            
        spot=np.argmax(AUCs_CV)  #place of optimal threshold for the marker after cross validation
        overall_optimals[TMs[mi]]=optimals[spot]    #optimal threshold for the marker after cross validation
    
    return overall_optimals

def find_nearest(array, value, pos):
    '''calculate the range of the threshold by taking into account the standard deviation of the max metric value'''
    array = np.asarray(array)
    diff = array - value  #value to consider
    top=diff[pos:]   #threshold values above maximum
    bot=diff[:pos]   #threshold values below maximum
    top_idx=top.argmin()+pos  #position where metric value is equal to max metric minus its std 
    bot_idx=bot.argmin()   #position where metric value is equal to max metric minus its std 
    return bot_idx,top_idx

def optimal_thresBootstrap(dframe,category='lung_carcinoma',used_metric='AUC'):
    '''determine the optimal threshold for each marker by applying Bootstrap and optimalization of the chosen metric'''
    dframe, kept=remove_nan_dframe(dframe,category)  #remove Nans
    (rows,columns)=dframe.shape
    TMs=dframe.columns[6:13]    #names of all tumor markers
    threshold=np.linspace(0,150,400)   #define possible thresholds
    labels=['No','Yes']
    lut = dict(zip(labels, [0,1]))
    y_true=dframe[category].map(lut)    #map the true classification to binary values
    k=5   #number of times boorstrap is applied
    selection=dframe.index.tolist()   #the indexes which can be selected to use
    optimal_range=dict()   #the optimal range for each threshold
    optimal_means=dict()   #the threshold value with highest mean
    
    for mi,marker in enumerate(range(6,13)):  #look at each marker separately
        metric=np.zeros((len(threshold),k))   #make room in memory for the AUCs
        for i in range(0,k):   #applying Bootstrap multiple times
            ti = [randint(0, len(dframe[TMs[mi]])-1) for p in range(0, len(dframe[TMs[mi]]))]   #select random indices
            train_index=[selection[z] for z in ti]  #select the indexes to be used which are present in de dataframe
            for index,thres in enumerate(threshold):   #loop over all of the possible threshold values
                LC_result=np.zeros(len(train_index))  #make room in memory for classification
                y_res=np.zeros(len(train_index))  #the true results for this Bootstrap round
                for ind,f_idx in enumerate(train_index):   #look at each selected index 
                    if (dframe.loc[f_idx,TMs[mi]])>=thres:   #classification process
                        LC_result[ind]=1
                    y_res[ind]=y_true.loc[f_idx]  #correct classificaton accompanied with this selected index
                if used_metric=='AUC':  
                    fpr, tpr, _ = roc_curve(y_res, LC_result)   #determine roc of each threshold
                    metric[index,i]=auc(fpr, tpr)   #determine AUC of each threshold
                elif used_metric=='F1':
                    metric[index,i]=f1_score(y_res,LC_result)  #F1 score
                elif used_metric=='precision':
                    metric[index,i]=precision_score(y_res,LC_result)  #precision score
        means=np.mean(metric,axis=1)  #calculate means of the metric
        stand=np.std(metric,axis=1)  #std of metric
        #plot result for each individual marker
        plt.errorbar(threshold,means,yerr=stand,linestyle='-',ecolor='black')  
        label=TMs[mi].split(' ')
        plt.xlabel('Threshold value '+label[1])
        plt.ylabel(used_metric)
        plt.title('Threshold values versus '+used_metric+ ' with Bootstrap method for the tumor marker: '+ label[0])
        plt.show()
        
        if used_metric=='AUC':
            spot=np.argmax(means)  #place with highest mean metric score
            t_range=means[spot]-np.abs(stand[spot])  #highest mean minus its standard deviation
            bot,top=find_nearest(means,t_range,spot)  #threshold indexes which match the calculated value
            string='-'.join([str(threshold[bot]),str(threshold[top])])  #range written in a string
            optimal_range[TMs[mi]]=string  #add range to dict
            optimal_means[TMs[mi]]=threshold[spot]   #add best threshold considering mean metric to dict
        
        elif used_metric=='F1':
            spot=np.argmax(means)  #place with highest mean metric score
            optimal_means[TMs[mi]]=threshold[spot] #add best threshold considering mean metric to dict
        elif used_metric=='precision':
            means=np.where(means>0.98,0,means) #every metric value which is to high to be considered as real/realistic is set to 0 
            spot=np.argmax(means) #place with highest mean metric score
            optimal_means[TMs[mi]]=threshold[spot] #add best threshold considering mean metric to dict
            
    return optimal_range,optimal_means
    
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
    extra=True  #provide additional data to the classifiers of age and smoking history
    if extra==True: #remove the Nan's for the ages and smoking history if data will have to be included
        dframe,_=remove_nan_int(dframe,'age')
        dframe,_=remove_nan_dframe(dframe,'smoking_history')
    y_true=dframe[cat]
    labels=y_true.unique()  #determine unique labels
    length=range(0,len(labels))  #provide a integer to each label
    lut = dict(zip(labels, length)) #create dictionary of possible options
    markers=dframe.iloc[:,6:13] #TM
    TMs=dframe.columns[6:13]
    y_true=y_true.map(lut)   #convert each string to the corresponding integer in the dictionary
        
    if normalize==True and smote!=True:   #scale each of the columns of the tumor markers
        markers = pd.DataFrame(preprocessing.scale(markers.values,axis=0),columns=markers.columns)
        
    if extra==True:
        ages=dframe['age']
        ages=np.rint(ages)  #round the ages to the nearest integer
        markers['age'] = ages #add the ages to the dataframe with the tumor markers
        
        smoking=dframe['smoking_history'] 
        transf={'Nooit':0,'Verleden':1,'Actief':2}   #dictonary to transform the strings to integers
        smoking=smoking.map(transf)   #map the strings in the list with the provided dictionary
        markers['smoking_history'] = smoking  #add the smoking history to the dataframe with the tumor markers
        
    X_train, X_test, y_train, y_test = train_test_split(markers.values, y_true, test_size=0.2)   #split the data in a training set and a test set
    
    if smote==True:     #apply synthetic Minority Over-sampling if specified (usually for skewed data distribution)
        sm = SMOTE(random_state=42)   #initialization
        name=markers.columns   #names of the TM's
        X_train,y_train=sm.fit_resample(X_train,y_train)  #apply operation and provide new data
        X_train=pd.DataFrame(X_train,columns=name)   #convert the TM list to a Dataframe
        
    if normalize==True and smote==True:   #scale each of the columns of the tumor markers
        X_test = pd.DataFrame(X_test, columns=markers.columns)
        markers[TMs] = preprocessing.scale(markers.values[:,0:7],axis=0)
        X_train[TMs] = preprocessing.scale(X_train.values[:,0:7],axis=0)
        X_test[TMs] = preprocessing.scale(X_test.values[:,0:7],axis=0)
    
    return markers, y_true, X_train, X_test, y_train, y_test, labels, lut

def det_CVscore(clf,markers,y_true):
    '''apply cross validation score and determine the mean and standard deviation of the score'''
    score=cross_val_score(clf,markers,y_true,cv=10,scoring='roc_auc')  #cross validation step
    score_f1=cross_val_score(clf,markers,y_true,cv=10,scoring='f1')
    CV_score={'mean AUC':np.mean(score),'std AUC':np.std(score),'mean F1':np.mean(score_f1),'std F1':np.std(score_f1)}
    
    return CV_score
    
def decisionT(dframe,cat,save_roc):
    '''Set up a decision tree classifier and train it after which predictions are made for the test set and statistics for this classification are calculated'''
    markers, y_true, X_train, X_test, y_train, y_test, labels, lut=prepare_data(dframe,cat,normalize=False,smote=True) #prepare the data
    clf = tree.DecisionTreeClassifier() #initialization of the classifier
    clf.fit(X_train,y_train)  #fit classifier to training data
    #visualize_DT(clf,dframe.columns[6:13],labels)

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
file_loc='data_new.csv'
dframe=read_data(file_loc)    #read data
dframe=remove_nan_markers(dframe)

make_clustermap(dframe=dframe, remove=True, save_fig=False, class_sort=category_to_investigate)

thresholds={'TM_CA15.3 (U/mL)': 35,'TM_CEA (ng/mL)':5,'TM_CYFRA (ng/mL)':3.3,'TM_NSE (ng/mL)':25,'TM_PROGRP (pg/mL)':50,'TM_SCC (ng/mL)':2}
if category_to_investigate=='lung_carcinoma':
    PPV_p,NPV_p,sensi_p,speci_p,report_p=approach_paper(dframe,thresholds,category_to_investigate)        
aucDT,PPV_DT,NPV_DT,sensitivity_DT,specificity_DT, report_DT, CV_score_DT=decisionT(dframe,category_to_investigate,save_roc=False)
aucLC,PPV_LC,NPV_LC,sensitivity_LC,specificity_LC, report_LC, CV_score_LC=Logistic_clas(dframe,category_to_investigate,save_roc=False)
