# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 18:24:48 2019
@author: shamilsons
Prediction of Chronic Kidney Diseases by using ML algorithms
The ML algorithms (KNN, NB, DT, LR, ANN, SVM, RF)
"""
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from random import *
import matplotlib.pyplot as plt
import seaborn as sns
from string import *
import numpy as np
import pandas as pd

'''
Early stage of Indians Chronic Kidney Disease(CKD)
1.Age(numerical) - age in years
2.Blood Pressure(numerical) - bp in mm/Hg
3.Specific Gravity(nominal) - sg -> (1.005,1.010,1.015,1.020,1.025)
4.Albumin(nominal) - al -> (0,1,2,3,4,5)
5.Sugar(nominal) - su -> (0,1,2,3,4,5)
6.Red Blood Cells(nominal) - rbc -> (normal,abnormal)
7.Pus Cell (nominal) - pc -> (normal,abnormal)
8.Pus Cell clumps(nominal) - pcc -> (present,notpresent)
9.Bacteria(nominal) - ba  -> (present,notpresent)
10.Blood Glucose Random(numerical) - bgr in mgs/dl
11.Blood Urea(numerical) - bu in mgs/dl
12.Serum Creatinine(numerical) - sc in mgs/dl
13.Sodium(numerical) - sod in mEq/L
14.Potassium(numerical) - pot in mEq/L
15.Hemoglobin(numerical) - hemo in gms
16.Packed  Cell Volume(numerical)
17.White Blood Cell Count(numerical) - wc in cells/cumm
18.Red Blood Cell Count(numerical) - rc in millions/cmm
19.Hypertension(nominal) - htn -> (yes,no)
20.Diabetes Mellitus(nominal) - dm -> (yes,no)
21.Coronary Artery Disease(nominal) - cad -> (yes,no)
22.Appetite(nominal) - appet -> (good,poor)
23.Pedal Edema(nominal) - pe -> (yes,no)
24.Anemia(nominal) - ane -> (yes,no)
25.Class (nominal) - class -> (ckd,notckd)

Number of Instances:  400 (250 CKD, 150 notckd)
Number of Attributes: 24 + class = 25 ( 11  numeric ,14  nominal)
'''

#Function to generate correlation HeatMap
def heatMap(df, mirror):
    corr = df.corr(method='spearman')
    #print(corr)
    
    #Plot figsize
    fig, ax = plt.subplots(figsize=(15,15))
    #Generate Color Map, red & blue
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    
    if mirror == True:
        #Generate Heat Map, allow annotations and place floats in map
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
        #Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns);
        #Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
    else:
        # Drop self-correlations
        dropSelf = np.zeros_like(corr)
        dropSelf[np.triu_indices_from(dropSelf)] = True# Generate Color Map
        colormap = sns.diverging_palette(220, 10, as_cmap=True)
        # Generate Heat Map, allow annotations and place floats in map
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
        # Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns);
        # Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
    
    #show plot
    plt.show()
    
#Read pima dataset file and extract instances
dataset = pd.read_csv('ckd_processed.csv')
#print (dataset.head())

#dataset['age'] = dataset['age'].map({'?':'NaN'})
#dataset['bp'] = dataset['bp'].map({'?':'NaN'})
#dataset['sg'] = dataset['sg'].map({'?':'NaN'})
#dataset['al'] = dataset['al'].map({'?':'NaN'})
#dataset['su'] = dataset['su'].map({'?':'NaN'})
dataset['rbc'] = dataset['rbc'].map({'normal': 2, 'abnormal': 1, '?':'NaN'})
dataset['pc'] = dataset['pc'].map({'normal': 2, 'abnormal': 1, '?':'NaN'})
dataset['pcc'] = dataset['pcc'].map({'present': 2, 'notpresent': 1, '?':'NaN'})
dataset['ba'] = dataset['ba'].map({'present': 2, 'notpresent': 1, '?':'NaN'})
#dataset['bgr'] = dataset['bgr'].map({'?':'NaN'})
#dataset['bu'] = dataset['bu'].map({'?':'NaN'})
#dataset['sc'] = dataset['sc'].map({'?':'NaN'})
#dataset['sod'] = dataset['sod'].map({'?':'NaN'})
#dataset['pot'] = dataset['pot'].map({'?':'NaN'})
#dataset['hemo'] = dataset['hemo'].map({'?':'NaN'})
#dataset['pcv'] = dataset['pcv'].map({'?':'NaN'})
#dataset['wbcc'] = dataset['wbcc'].map({'?':'NaN'})
#dataset['rbcc'] = dataset['rbcc'].map({'?':'NaN'})
dataset['htn'] = dataset['htn'].map({'yes': 2, 'no': 1, '?':'NaN'})
dataset['dm'] = dataset['dm'].map({'yes': 2, 'no': 1, '?':'NaN'})
dataset['cad'] = dataset['cad'].map({'yes': 2, 'no': 1, '?':'NaN'})
dataset['appet'] = dataset['appet'].map({'good': 2, 'poor': 1, '?':'NaN'})
dataset['pe'] = dataset['pe'].map({'yes': 2, 'no': 1, '?':'NaN'})
dataset['ane'] = dataset['ane'].map({'yes': 2, 'no': 1, '?':'NaN'})
dataset['class'] = dataset['class'].map({'ckd': 1, 'notckd': 0})

#Make all header values uppercase
dataset.columns = [hdr.upper() for hdr in dataset.columns]
#print (dataset.columns.values)

#print dataset.columns
#print dataset['class'].head(10)

X = dataset.iloc[:, 0:24].values
y = dataset.iloc[:, 24].values

#Handling missing values in the dataset by using mean strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,0:24])
X[:,0:24] = imputer.transform(X[:,0:24])

#print (type(X))
#print (type(y))
#print (X[:,0])
#print (y)

#Perform spearman correlational analysis 
#Create correlation matrix between features
feature_titles = ['AGE','BP','SG','AL','SU','RBC','PC','PCC','BA','BGR','BU','SC','SOD','POT','HEMO','PCV','WBCC','RBCC','HTN','DM','CAD','APPET','PE','ANE']

#Create only dateset of features without target variable for heatmap analysis
#tempDF = pd.DataFrame(data=np.float64(X), columns=feature_titles)
#print (tempDF.head(5))
#print (type(tempDF['AGE'][0]))
#Generate correlational HeatMap    
#heatMap(tempDF, False)

#Correlational analysis (non-linear) between Target and remaining features
'''
from scipy.stats import spearmanr
#print(X[:,1])
for idx in range(len(feature_titles)):
    spr_corr=float(spearmanr(y,X[:,idx])[0])
    trs_lower=-0.25
    trs_upper=0.25
    if(spr_corr>=trs_lower and spr_corr<=trs_upper):
        print ("Spearman corr. between TRG and "+feature_titles[idx]+":"+str(spearmanr(y,X[:,idx])[0]))
        #Based on treshold: 7-PCC:0.221, 8-BA:0.125, 13-POT:0.032, 16-WBCC:0.208, 20-CAD:0.210     
'''

#Features selection on indexes of features based on spearman correlational analysis
X_ftr_slc = X[:,[7,8,13,16,20]]
#print(np.shape(X_ftr_slc))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ftr_slc, y, stratify=y, random_state = 42)
#print X_train.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def performance_metrics(clr, y_test, y_pred):
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print("Sensitivity (recall):%.2f"%round(TPR[0],2))
    
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print("Specificity:%.2f"%round(TNR[0],2))
    
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print("Precision:%.2f"%round(PPV[0],2))
    
    # Negative predictive value
    NPV = TN/(TN+FN)
    print("Negative Predictive Value:{0:.2f}".format(NPV[0]))
    
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print("False positive rate:%.2f"%round(FPR[0],2))
    
    # False negative rate
    FNR = FN/(TP+FN)
    print("False negative rate:%.2f"%round(FNR[0],2))
    
    # False discovery rate
    FDR = FP/(TP+FP)
    print("False discovery rate:%.2f"%round(FDR[0],2))
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print("Overall Accuracy:%.2f"%round(ACC[0],2))
    
    #print (cm)

    print ('\n==== PREDICTION ACCURACIES OF ALGORITHMS =====\n')
    print ('Classifier Accuracy :',metrics.accuracy_score(y_test,y_pred),'--- MSE:',metrics.mean_squared_error(y_test,y_pred))
    print ("Train score :%.2f" %round(clr.score(X_train, y_train),2))
    print ("Test  score :%.2f" %round(clr.score(X_test, y_test),2))
    print("Classification report:", metrics.classification_report(y_test,y_pred))


# Fitting SVM to the Training set
svm_clr = SVC(C=1.0, kernel = 'poly', degree=2, random_state=42)
svm_clr.fit(X_train, y_train)

# Predicting the Test set results
y_pred = svm_clr.predict(X_test)

performance_metrics(svm_clr, y_test, y_pred)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(svm_clr, X_train, y_train, cv=10)
print(scores)
print(scores.mean())
#model_set = ['KNN', 'NB', 'DT', 'LR', 'ANN', 'SVM', 'RF']

