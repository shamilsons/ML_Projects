# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:35:09 2019
@author: Shahriar ShamilUulu 
Applying K-Nearest Neighbour classifier over iris dataset
"""
#import needed important libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#import scikit-learn libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#importing iris dataset 
iris=datasets.load_iris()
#iris.data has features, iris.target has labels
#print(iris.data)
#print(iris.target)

#putting iris data into pandas dataframe (similar like Excel)
data_iris = pd.DataFrame(data= np.c_[iris['data'], iris['target'], iris['target']], columns= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target', 'target_names'])

data_iris['target_names'] = data_iris['target_names'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

#check and drop columns with missing values 
print(data_iris.isnull().sum())
data_iris.dropna(axis=1, inplace=True)
#interpolate missing values
#data_iris.interpolate()

#print first 5 rows
print(data_iris.head())
#get number of rows and columns
print(data_iris.shape)
#get more information about dataset and columns
print(data_iris.info())
          
#explaratory data analysis
#perform correlational analysis
iris_corr_pearson = data_iris.corr()
iris_corr_spearman = data_iris.corr('spearman')
iris_corr_kendall = data_iris.corr('kendall')

#summary statistics on numerical data
summary_stats = data_iris.describe()

#scatter plots and histograms for the columns 
sns.pairplot(data_iris)

#heatnap based on correlational matrix for the dataset
sns.heatmap(data_iris.corr(), annot=True)
plt.plot()

#scatter-plot for two variables for three classes 
sns.FacetGrid(data_iris, hue="target_names").map(plt.scatter, "sepal_length", "sepal_width").add_legend()
plt.show()

#scatter-plot for two variables for three classes
sns.FacetGrid(data_iris, hue="target_names").map(plt.scatter, "petal_length", "petal_width").add_legend()
plt.show()

#box-plot for all independent numerical variables
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
sns.boxplot(x="target_names", y="sepal_length", data=data_iris)
plt.subplot(2,2,2)
sns.boxplot(x="target_names", y="sepal_width", data=data_iris)
plt.subplot(2,2,3)
sns.boxplot(x="target_names", y="petal_length", data=data_iris)
plt.subplot(2,2,4)
sns.boxplot(x="target_names", y="petal_width", data=data_iris)
plt.show()          

#Each entry has 4 attributes, 
#Spliting data into training and testing sets (hold-out method)
x_train, x_test, y_train, y_test=train_test_split(iris.data, iris.target, test_size=0.35, stratify=iris.target, random_state=42)

#instatiating the KNN classifier 
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(x_train, y_train)

print ("Train accuracy score : %.2f" %round(clf_knn.score(x_train, y_train)*100,2))

#calculate the prediction accuracy metric on test dataset
print("Test accuracy score : %.2f\n"%round(accuracy_score(y_test, clf_knn.predict(x_test))*100,2))

#we got accuracy of 96 when we use k=3, now lets try to plot graph with different k values and accuracy
#we now iterate our classifier and init it different k values and find accuracy
#accuracy values is 2D array, where each entry is [K, accuracy]
accuracy_values=[]

for x in range(1, x_train.shape[0]):
    clf_knn = KNeighborsClassifier(n_neighbors=x)
    clf_knn.fit(x_train,y_train)
    accuracy = accuracy_score(y_test, clf_knn.predict(x_test))
    accuracy_values.append([x, accuracy*100])
    pass

#converting normal python array to numpy array
accuracy_values=np.array(accuracy_values)

plt.plot(accuracy_values[:,0], accuracy_values[:,1])
plt.xlabel("Number of K-Neighbours")
plt.ylabel("Testing Accuracy")
plt.show()
