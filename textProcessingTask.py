# -*- coding: utf-8 -*-
'''
This program developed by Shahriar Shamiluulu, shahriar.shamiluulu@sdu.edu.kz
Computer Science Department - Suleyman Demirel University, Kaskelen, Kazakhstan 2017

The program used to perform sentimental analysis in three classes (positive, negative, neutral)
    The train.json file contains  8263 instances with features (id, text, sentiment)
    The test.json contains 2xxx instances with features (id, text)
    On total 9 different classification algorithms are used i.e.,
       (Logistic regression, Neural networks, Random forest, knn, Multinomial naive-base,
       Support vector machines, Decisin trees [CART], Adaptive Boosting, Naive-base)
       The best classification performance showed Logistic regression with accuracy: 66.8%

Platform background where the program has been developed
    Windows 7 SP2, Python 2.7, scikit-learn 0.18.1, nltk 3.2.2, matplotlib 2.0, numpy+mkl 1.12.1
    pandas 0.19.2, scipy 0.19, ...
'''

import string
import json
import nltk
import time
import math
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#removing stop_words from the text
stop_words = stopwords.words('russian')
stop_words.extend([u'что', u'это', u'так', u'вот', u'быть', u'как', u'в', u'к',
                   u'по',u'на',u'до',u'ао',u'рк', u'я', u'а', u'да', u'но', u'тебе',
                   u'мне', u'ты', u'и', u'у', u'на', u'ща', u'ага', u'так', u'там', u'какие',
                   u'который', u'какая', u'туда', u'давай', u'короче', u'кажется', u'вообще', u'ну', u'не', u'чет',
                   u'неа', u'свои', u'наше', u'хотя', u'такое', u'например', u'кароч', u'как-то', u'нам', u'хм', u'всем',
                   u'нет', u'да', u'оно', u'своем', u'про', u'вы', u'м', u'тд', u'вся', u'кто-то', u'что-то', u'вам',
                   u'это', u'эта', u'эти', u'этот', u'прям', u'либо', u'как', u'мы', u'просто', u'блин', u'очень', u'самые',
                   u'твоем', u'ваша', u'кстати', u'вроде', u'типа', u'пока', u'ок', '\'\'', '``', '\n'])

def readDataFileJSON(file_name, file_type=1):
    # reading all data from (train.json or test.json)into one nested-multidimentational list [print type(json_data), print len(json_data)]
    # type: 1 (for train data)
    print "2. Reading json file ..."

    if(file_type==1):
        with open(file_name) as json_file:
            data_from_file = json.load(json_file)
        return data_from_file
    else:
        with open(file_name) as json_file:
            data_from_file = json.load(json_file)
            print "Length of test file:", len(data_from_file)
        return data_from_file

def dataTrainTextPreprocessing(data_from_file, no_of_instances, no_of_features, break_check=True, store_processed_data=1):
    print "3. Data train text preprocessing process started ..."

    word_dictionary = []
    word_frequency_set = []
    train_set = []
    record_no=no_of_instances

    # Takes the world and convert to original word form like: houses, houses10 etc is converted to house
    stemmer = nltk.stem.snowball.RussianStemmer('russian')

    # shows how many text records has been processed so far
    stmt_counter = 1

    for snt in data_from_file:
        # print snt['id'],"---",snt['sentiment']

        # 1. TEXT PRE-PROCESSING STEP
        # divide sentance into words
        tokens = (word_tokenize(snt['text']))

        # delete punctuation symbols and convert words into lowercase
        tokens = [i.lower() for i in tokens if (i not in string.punctuation)]

        # remove all stopwords
        tokens = [i for i in tokens if (i not in stop_words)]

        # cleaning words
        tokens = [i.replace(u"«", "").replace(u"»", "") for i in tokens]
        tokens = [i.replace(u"—", "") for i in tokens]
        tokens = [i.replace(u"-", "") for i in tokens]
        tokens = [i.replace(u"ctrl+enter", "") for i in tokens]
        tokens = [i.replace(u"№3/4", "") for i in tokens]
        tokens = [i.replace(u"№22и", "") for i in tokens]
        tokens = [i.replace(u"allur", "") for i in tokens]
        tokens = [i.replace(u"alluraut", "") for i in tokens]
        tokens = [i.replace(u"+item.suffix", "") for i in tokens]
        tokens = [i.replace(u"+0,4", "") for i in tokens]
        tokens = [i.replace(u"utc+6", "") for i in tokens]
       # print tokens

        tokens_updated = []
        for w in tokens:
            #The word must more than 3 letters, it must not be digit or float number
            if (len(w) > 3 and w.isdigit() == False and w[0].isdigit()==False):
                stmword = stemmer.stem(w)
                # print str(count) + ": " + w + " : " + stmword
                tokens_updated.append(stmword)
                word_frequency_set.append(stmword)
                if stmword not in word_dictionary:
                    word_dictionary.append(stmword)
                    # count = count + 1

        train_set.append([snt['id'], tokens_updated, snt['sentiment']])
        print stmt_counter, " record with id:", snt['id']," and sentiment:",snt['sentiment']," processed"
        # print stmt_counter,". preprocessing of text process finished for:",snt['id'],' class:',snt['sentiment']
        # print stmt_counter,". preprocessing of text process finished for:", snt['id']

        stmt_counter += 1
        no_of_instances -= 1
        if (no_of_instances == 0 and break_check == True):
            if(store_processed_data==1):
                #Store train processed set for later use
                print "Storing processed dataset ..."
                with open("train_set_"+str(record_no)+"_"+str(no_of_features)+".txt", "wb") as fp:
                    pickle.dump(train_set, fp)

                # Store word frequency set
                with open("word_frequency_set_"+str(record_no)+"_"+str(no_of_features)+".txt", "wb") as fp:
                    pickle.dump(word_frequency_set, fp)

                '''
                # Code to read stored train set
                with open("train_set_full_1.txt", "rb") as fp:
                    train_set = pickle.load(fp)
                fp.close()

                with open("word_frequency_set.txt", "rb") as fp:
                    word_frequency_set = pickle.load(fp)
                fp.close()
                '''
                return train_set, word_frequency_set, word_dictionary
            break

def dataTestTextPreprocessing(data_from_file, no_of_instances, no_of_features, break_check=True, store_processed_data=1):
    print "3. Data test text preprocessing process started ..."

    word_dictionary = []
    word_frequency_set = []
    test_set = []
    record_no=no_of_instances

    # Takes the world and convert to original word form like: houses, houses10 etc is converted to house
    stemmer = nltk.stem.snowball.RussianStemmer('russian')

    # shows how many text records has been processed so far
    stmt_counter = 1

    for snt in data_from_file:
        # print snt['id'],"---",snt['sentiment']

        # 1. TEXT PRE-PROCESSING STEP
        # divide sentance into words
        tokens = (word_tokenize(snt['text']))

        # delete punctuation symbols and convert words into lowercase
        tokens = [i.lower() for i in tokens if (i not in string.punctuation)]

        # remove all stopwords
        tokens = [i for i in tokens if (i not in stop_words)]

        # cleaning words
        tokens = [i.replace(u"«", "").replace(u"»", "") for i in tokens]
        tokens = [i.replace(u"—", "") for i in tokens]
        tokens = [i.replace(u"-", "") for i in tokens]
        tokens = [i.replace(u"ctrl+enter", "") for i in tokens]
        tokens = [i.replace(u"№3/4", "") for i in tokens]
        tokens = [i.replace(u"№22и", "") for i in tokens]
        tokens = [i.replace(u"allur", "") for i in tokens]
        tokens = [i.replace(u"alluraut", "") for i in tokens]
        tokens = [i.replace(u"+item.suffix", "") for i in tokens]
        tokens = [i.replace(u"+0,4", "") for i in tokens]
        tokens = [i.replace(u"utc+6", "") for i in tokens]
       # print tokens

        tokens_updated = []
        for w in tokens:
            #The word must more than 3 letters, it must not be digit or float number
            if (len(w) > 3 and w.isdigit() == False and w[0].isdigit()==False):
                stmword = stemmer.stem(w)
                # print str(count) + ": " + w + " : " + stmword
                tokens_updated.append(stmword)
                word_frequency_set.append(stmword)
                if stmword not in word_dictionary:
                    word_dictionary.append(stmword)
                    # count = count + 1

        test_set.append([snt['id'], tokens_updated])
        print stmt_counter, " record with id:", snt['id']," processed"
        # print stmt_counter,". preprocessing of text process finished for:",snt['id'],' class:',snt['sentiment']
        # print stmt_counter,". preprocessing of text process finished for:", snt['id']

        stmt_counter += 1
        no_of_instances -= 1
        if (no_of_instances == 0 and break_check == True):
            if(store_processed_data==1):
                #Store train processed set for later use
                print "Storing processed dataset ..."
                with open("test_set_"+str(record_no)+"_"+str(no_of_features)+".txt", "wb") as fp:
                    pickle.dump(test_set, fp)

                # Store word frequency set
                with open("word_frequency_test_set_"+str(record_no)+"_"+str(no_of_features)+".txt", "wb") as fp:
                    pickle.dump(word_frequency_set, fp)

                '''
                # Code to read stored train set
                with open("train_set_full_1.txt", "rb") as fp:
                    train_set = pickle.load(fp)
                fp.close()

                with open("word_frequency_set.txt", "rb") as fp:
                    word_frequency_set = pickle.load(fp)
                fp.close()
                '''
                #return test_set, word_frequency_set, word_dictionary
                return test_set
            break

def processingTrainWordFrequencies(no_of_features, word_frequency_set, train_set):
    print "4. Processing train word list frequencies ..."

    # Get frequencies of each word in the form of word - number of times it appears
    # Most common 5000 features according to frequency appearence are included
    all_words = nltk.FreqDist(word_frequency_set)
    dataset_updated = []

    for record in train_set:
        record_words_frequency_bag = []
        for word in all_words.most_common(no_of_features):
            # print word[0]+" "+str(word[1])
            # Create frequency bag for each instance
            record_words_frequency_bag.append(record[1].count(word[0]))

        # Make classes parametric (1-negative, 2-neutral, 3-positive)
        # Because many algorithms work with numerical values and perfom classification accordingly
        if (record[2] == u'negative'):
            class_var = 1
        elif (record[2] == u'neutral'):
            class_var = 2
        else:
            class_var = 3

        # Storing preprocessed dataset in the form of id, class, frequency of words
        # Ex: [1945, 1, [2, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
        dataset_updated.append([record[0], class_var, record_words_frequency_bag])

    return dataset_updated

def processingTestWordFrequencies(no_of_features, word_frequency_set, test_set):
    print "4. Processing test word list frequencies ..."

    # Get frequencies of each word in the form of word - number of times it appears
    # Most common 5000 features according to frequency appearence are included
    all_words = nltk.FreqDist(word_frequency_set)
    dataset_updated = []

    for record in test_set:
        record_words_frequency_bag = []
        for word in all_words.most_common(no_of_features):
            # print word[0]+" "+str(word[1])
            # Create frequency bag for each instance
            record_words_frequency_bag.append(record[1].count(word[0]))

        # Storing preprocessed dataset in the form of id, frequency of words
        # Ex: [1945, [2, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
        dataset_updated.append([record[0], record_words_frequency_bag])

    return dataset_updated

def classificationProcess(dataset_train, dataset_test, classifiers):
    print "5. Starting Classification Process ..."

    # Perform splitting process of features and classes
    data = []     # holds instance features
    target = []   # holds classes for each instance
    for record in dataset_train:
        target.append(record[1])
        data.append(record[2])

    # Creating hold-out method by dividing dataset into Train:75% and Test:25%
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.25, random_state=0)
    expectedClass = Y_test

    print '\n==== CLASSIFICATION ACCURACIES OF ALGORITHMS ====='

    if('knn' in classifiers):
        #creating the instance of KNN classifier
        knn=KNeighborsClassifier(n_neighbors=5).fit(X_train,Y_train)
        predictedClassKNN = knn.predict(X_test)
        print "\nClassification Report for K-Nearest Neighbout (KNN)"
        print(metrics.classification_report(expectedClass,predictedClassKNN))
        print "Confusion matrix for KNN"
        print(metrics.confusion_matrix(expectedClass, predictedClassKNN))
        print "\nAccuracy average score:", "%0.1f"%(metrics.accuracy_score(expectedClass,predictedClassKNN)*100),'%'
        print "RMSE Error:", "%0.2f"%(math.sqrt(metrics.mean_squared_error(expectedClass,predictedClassKNN)))
        #print "Homogeneity score:", "%0.2f"%(metrics.homogeneity_score(expectedClass,predictedClassKNN))
        del(knn)

    if('dt' in classifiers):
        dt=DecisionTreeClassifier().fit(X_train,Y_train)
        predictedClassDT = dt.predict(X_test)
        print "\nClassification Report for Decision-tree (DT)"
        print(metrics.classification_report(expectedClass,predictedClassDT))
        print "Confusion matrix for DT"
        print(metrics.confusion_matrix(expectedClass, predictedClassDT))
        print "\nAccuracy average score:", "%0.1f"%(metrics.accuracy_score(expectedClass,predictedClassDT)*100),'%'
        print "RMSE Error:", "%0.2f"%(math.sqrt(metrics.mean_squared_error(expectedClass,predictedClassDT)))
        #print "Homogeneity score:", "%0.2f"%(metrics.homogeneity_score(expectedClass,predictedClassDT))
        del(dt)

    if('nb' in classifiers):
        nb = GaussianNB().fit(X_train,Y_train)
        predictedClassNB = nb.predict(X_test)
        print "\nClassification Report for Naive-base (NB)"
        print(metrics.classification_report(expectedClass,predictedClassNB))
        print "Confusion matrix for Naive-base"
        print(metrics.confusion_matrix(expectedClass, predictedClassNB))
        print "\nAccuracy average score:", "%0.1f"%(metrics.accuracy_score(expectedClass,predictedClassNB)*100),'%'
        print "RMSE Error:", "%0.2f"%(math.sqrt(metrics.mean_squared_error(expectedClass,predictedClassNB)))
        #print "Homogeneity score:", "%0.2f"%(metrics.homogeneity_score(expectedClass,predictedClassNB))
        del(nb)

    if('lr' in classifiers):
        #creating the instance of Logistic Regression(Logit) classifier
        lr = LogisticRegression().fit(X_train,Y_train)
        predictedClassLR = lr.predict(X_test)
        print "\nClassification Report for Logistic Regression (LR)"
        print(metrics.classification_report(expectedClass,predictedClassLR))
        print "Confusion matrix for LR"
        print(metrics.confusion_matrix(expectedClass, predictedClassLR))
        print "\nAccuracy average score:", "%0.1f"%(metrics.accuracy_score(expectedClass,predictedClassLR)*100),'%'
        print "RMSE Error:", "%0.2f"%(math.sqrt(metrics.mean_squared_error(expectedClass,predictedClassLR)))
        #print "Homogeneity score:", "%0.2f"%(metrics.homogeneity_score(expectedClass,predictedClassLR))

        pdcl=[]
        for test_record in dataset_test:
            # Make classes parametric (1-negative, 2-neutral, 3-positive)
            #print  test_record
            prd_class=lr.predict([test_record[1]])
            if(prd_class==1):
                #print "id:", test_record[0], " predicted: negative ", prd_class
                pdcl.append([test_record[0], 'negative'])
            elif(prd_class==2):
                #print "id:", test_record[0], " predicted: neutral ", prd_class
                pdcl.append([test_record[0], 'neutral'])
            else:
                #print "id:", test_record[0], " predicted: positive ", prd_class
                pdcl.append([test_record[0], 'positive'])
                #print "id:",test_record[0]," predicted:",lr.predict([test_record[1]]), " with probability:", lr.predict_proba([test_record[1]])


        import csv
        with open('predictedTestResults.csv', 'wb') as f:
            wtr = csv.writer(f, delimiter=',')
            wtr.writerow(['id', 'sentiment'])
            wtr.writerows(pdcl)
    del(lr)

    if('svm' in classifiers):
        #creating the instance of SVM(support vector machine) classifier
        svm=SVC().fit(X_train,Y_train)
        predictedClassSVM = svm.predict(X_test)
        print "\nClassification Report for (SVM)"
        print(metrics.classification_report(expectedClass,predictedClassSVM))
        print "Confusion matrix for SVM"
        print(metrics.confusion_matrix(expectedClass, predictedClassSVM))
        print "\nAccuracy average score:", "%0.1f"%(metrics.accuracy_score(expectedClass,predictedClassSVM)*100),'%'
        print "RMSE Error:", "%0.2f"%(math.sqrt(metrics.mean_squared_error(expectedClass,predictedClassSVM)))
        #print "Homogeneity score:", "%0.2f"%(metrics.homogeneity_score(expectedClass,predictedClassSVM))
        del(svm)

    if('abt' in classifiers):
        #creating the instance of ABT(adaboosting algorithm) classifier
        abt = AdaBoostClassifier(n_estimators=40, random_state=3).fit(X_train,Y_train)
        predictedClassABT = abt.predict(X_test)
        print "\nClassification Report for AdaBoostClassifier (ABT)"
        print(metrics.classification_report(expectedClass,predictedClassABT))
        print "Confusion matrix for ABT"
        print(metrics.confusion_matrix(expectedClass, predictedClassABT))
        print "\nAccuracy average score:", "%0.1f"%(metrics.accuracy_score(expectedClass,predictedClassABT)*100),'%'
        print "RMSE Error:", "%0.2f"%(math.sqrt(metrics.mean_squared_error(expectedClass,predictedClassABT)))
        #print "Homogeneity score:", "%0.2f"%(metrics.homogeneity_score(expectedClass,predictedClassABT))
        del(abt)

    if('ann' in classifiers):
        #creating the instance of ANN(neural network) classifier
        ann = MLPClassifier(activation='relu',solver='lbfgs', max_iter=400, hidden_layer_sizes=100).fit(X_train,Y_train)
        predictedClassANN = ann.predict(X_test)
        print "\nClassification Report for Artificial Neural Network (ANN)"
        print(metrics.classification_report(expectedClass,predictedClassANN))
        print "Confusion matrix for ANN"
        print(metrics.confusion_matrix(expectedClass,predictedClassANN))
        print "\nAccuracy average score:", "%0.1f"%(metrics.accuracy_score(expectedClass, predictedClassANN)*100),'%'
        print "RMSE Error:", "%0.2f"%(math.sqrt(metrics.mean_squared_error(expectedClass,predictedClassANN)))
        #print "Homogeneity score:", "%0.2f"%(metrics.homogeneity_score(expectedClass,predictedClassANN))
        del(ann)

    if('mnb' in classifiers):
        mnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None).fit(X_train, Y_train)
        predictedClassMNB = mnb.predict(X_test)
        print "\nClassification Report for Multinomial Naive-Base (MNB)"
        print(metrics.classification_report(expectedClass, predictedClassMNB))
        print "Confusion matrix for MNB"
        print(metrics.confusion_matrix(expectedClass, predictedClassMNB))
        print "\nAccuracy average score:", "%0.1f"%(metrics.accuracy_score(expectedClass, predictedClassMNB)*100),'%'
        print "RMSE Error:", "%0.2f"%(math.sqrt(metrics.mean_squared_error(expectedClass, predictedClassMNB)))
        #print "Homogeneity score:", "%0.2f"%(metrics.homogeneity_score(expectedClass, predictedClassMNB))
        del (mnb)

    if('rfc' in classifiers):
        rfc = RandomForestClassifier(n_estimators=20, criterion='gini', n_jobs=2).fit(X_train, Y_train)
        predictedClassRFC = rfc.predict(X_test)
        print "\nClassification Report for RandomForestClassifier (RFC)"
        print(metrics.classification_report(expectedClass, predictedClassRFC))
        print "Confusion matrix for RFC"
        print(metrics.confusion_matrix(expectedClass, predictedClassRFC))
        print "\nAccuracy average score:", "%0.1f"%(metrics.accuracy_score(expectedClass, predictedClassRFC)*100),'%'
        print "RMSE Error:", "%0.2f"%(math.sqrt(metrics.mean_squared_error(expectedClass, predictedClassRFC)))
        #print "Homogeneity score:", "%0.2f"%(metrics.homogeneity_score(expectedClass, predictedClassRFC))
        del (rfc)

    if(len(classifiers)==0):
        print "Please select enter at least one classifier in order to build the classification model ..."

def main():
    print "1. Program execution started time set..."
    start_time = time.clock()

    no_of_instances=8263   # default value for instance number is 8263
    no_of_features = 5000  # default value for number of features is 5000
    store_processed_data=1 # whether to store processed data from list into the file for quicker use

    no_of_test_instances = 2056 # default test isntances is 2056

    # read data from json file, store in the list and return. File type 1 means read train dataset
    data_from_train_file = readDataFileJSON("train.json", file_type=1)
    data_from_test_file = readDataFileJSON("test.json", file_type=0)

    # The function returns train_set, word_frequency_set, word_dictionary
    train_set, word_frequency_set, word_dictionary = dataTrainTextPreprocessing(data_from_train_file, no_of_instances, no_of_features, True, store_processed_data)
    test_set = dataTestTextPreprocessing(data_from_test_file, no_of_test_instances, no_of_features, break_check=True, store_processed_data=1)

    # The function returns updated dataset with class_variable and word frequencies for each record
    dataset_updated_train = processingTrainWordFrequencies(no_of_features, word_frequency_set, train_set)
    dataset_updated_test = processingTestWordFrequencies(no_of_features, word_frequency_set, test_set)

    # Main function where classification process done. Contains algorithms like (KNN[k nearest neighbour], DT [decision tree],
    # NB [naive-base], LR [logistic regression], SVM [support vector machines], ABT [adaptive boosting], ANN [artificial neural network]
    # LDA [lineardiscirimantanalyzer], MNB [multinomial naive base], RFC [random forest classifier])
    # The best performance algorithms found RFC, MNB, LR, KNN
    # In order to choose classifier select one of these (knn, dt, nb, lr, svm, abt, ann, mnb, rfc)
    # In order to avoid errors like: "MemoryError" run classifiers one by one, because they take alot of memory to build the model
    chooseClassifiers=['lr']
    classificationProcess(dataset_updated_train, dataset_updated_test, chooseClassifiers)

    print "\n=== Program Execution Time ==="
    print "Total: ", time.clock() - start_time, " secs"

#Call main function to start program execution
main()

'''
==== CLASSIFICATION ACCURACIES OF ALGORITHMS =====

Number of instances:8263
Number of features: 5000
Algorithms: LR, ANN, RFC, KNN, MNB, SVM, DT, ABT, NB (ordered according to classification accuracies)

Classification Report for Logistic Regression (LR)
             precision    recall  f1-score   support

          1       0.65      0.63      0.64       362
          2       0.68      0.69      0.69      1003
          3       0.66      0.65      0.65       701

avg / total       0.67      0.67      0.67      2066

Confusion matrix for LR
[[229  98  35]
 [103 694 206]
 [ 18 225 458]]

Accuracy average score: 66.8%
RMSE Error: 0.64

Classification Report for Artificial Neural Network (ANN)
             precision    recall  f1-score   support

          1       0.67      0.64      0.66       362
          2       0.67      0.69      0.68      1003
          3       0.65      0.64      0.65       701

avg / total       0.66      0.67      0.66      2066

Confusion matrix for ANN
[[233 101  28]
 [102 693 208]
 [ 14 239 448]]

Accuracy average score: 66.5%
RMSE Error: 0.63


Classification Report for RandomForestClassifier (RFC)
             precision    recall  f1-score   support

          1       0.70      0.57      0.63       362
          2       0.64      0.76      0.70      1003
          3       0.69      0.57      0.62       701

avg / total       0.67      0.66      0.66      2066

Confusion matrix for RFC
[[207 134  21]
 [ 76 765 162]
 [ 11 291 399]]

Accuracy average score: 66.4%
RMSE Error: 0.62

Classification Report for (KNN)
             precision    recall  f1-score   support

          1       0.56      0.61      0.59       285
          2       0.68      0.70      0.69       806
          3       0.65      0.59      0.62       562

avg / total       0.65      0.65      0.65      1653

Confusion matrix for KNN
[[175  82  28]
 [ 94 561 151]
 [ 41 187 334]]


Accuracy average score: 65.2%
RMSE: 0.69


Classification Report for Multinomial Naive-Base (MNB)
             precision    recall  f1-score   support

          1       0.49      0.78      0.61       362
          2       0.76      0.50      0.60      1003
          3       0.64      0.76      0.69       701

avg / total       0.67      0.64      0.63      2066

Confusion matrix for MNB
[[282  50  30]
 [226 503 274]
 [ 62 108 531]]

Accuracy average score: 63.7%
RMSE Error: 0.70

Classification Report for Support Vector Machine (SVM)
             precision    recall  f1-score   support

          1       0.74      0.24      0.36       285
          2       0.58      0.87      0.70       806
          3       0.70      0.44      0.54       562

avg / total       0.65      0.61      0.58      1653

Confusion matrix for SVM
[[ 67 198  20]
 [ 21 702  83]
 [  3 314 245]]

Accuracy average score: 61.3%
RMSE: 0.66


Classification Report for Decision-tree (DT)
             precision    recall  f1-score   support

          1       0.54      0.57      0.55       285
          2       0.61      0.63      0.62       806
          3       0.57      0.53      0.55       562

avg / total       0.59      0.59      0.59      1653

Confusion matrix for Decision-tree
[[162  98  25]
 [100 509 197]
 [ 38 224 300]]


Accuracy average score: 59.4%
RMSE: 0.72

Classification Report for Adaptive Boosting (ABT)
             precision    recall  f1-score   support

          1       0.55      0.47      0.51       285
          2       0.56      0.66      0.60       806
          3       0.59      0.49      0.53       562

avg / total       0.57      0.57      0.56      1653

Confusion matrix for ABT
[[134 141  10]
 [ 96 529 181]
 [ 13 276 273]]


Accuracy average score: 57.6%
RMSE: 0.69


Classification Report for Naive-base (NB)
             precision    recall  f1-score   support

          1       0.31      0.88      0.45       285
          2       0.74      0.29      0.41       806
          3       0.62      0.57      0.60       562

avg / total       0.63      0.49      0.48      1653

Confusion matrix for Naive-base
[[251  27   7]
 [384 231 191]
 [187  53 322]]

Accuracy average score: 47.3%
MSE: 0.93


=== Execution End Time ===
3698.21180837  secs
'''