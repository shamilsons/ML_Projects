#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import json
import pymorphy2
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
 
morph = pymorphy2.MorphAnalyzer()
# parseds = morph.parse(u'?????')
 
with io.open('train.json','r',encoding="utf-8") as data_file:
    news = json.load(data_file)
    #print news[0]["text"].encode("utf-8")
    #print news[0]["sentiment"]
    news_collection = []
    sentiment_collection = []
    for i in range(0,1000):
        content = news[i]["text"]
        # print content
        tokens = [j.lower() for j in nltk.word_tokenize(content)]
        stemmed_text = ""
        for token in tokens:
            #print token.encode("utf-8"),
            parseds = morph.parse(token)
            # print parseds[0][2].encode("utf-8")
            stemmed_text = stemmed_text+" "+parseds[0][2]
        print i,stemmed_text + '\n'
        news_collection.append(stemmed_text)
        sentiment_collection.append(news[i]["sentiment"])
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(news_collection)
    #print X_train_counts.shape
    # #print count_vect.vocabulary_.get(u'????')
    #print X_train_counts[:,324]
    tf_transformer = TfidfTransformer()
    X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
    #print X_train_tfidf.shape
    # #print X_train_tfidf[:,324]
    new_collection = []
    new_sentiment_collection = []
    for i in range(1001,1200):
        content = news[i]["text"]
        tokens = [j.lower() for j in nltk.word_tokenize(content)]
        stemmed_text = ""
        for token in tokens:
            #print token.encode("utf-8"),\
            parseds = morph.parse(token)
            #print parseds[0][2].encode("utf-8")
            stemmed_text = stemmed_text+" "+parseds[0][2]
        #print stemmed_text
        new_collection.append(stemmed_text)
        new_sentiment_collection.append(news[i]["sentiment"])
    X_new_counts = count_vect.transform(new_collection)
    X_new_tfidf = tf_transformer.transform(X_new_counts)
    clf = MultinomialNB().fit(X_train_tfidf,sentiment_collection)
    predicted = clf.predict(X_new_tfidf)
    print predicted
    for text,category,actual_category in zip(new_collection,predicted,new_sentiment_collection):
        print category," ",actual_category," ",text[:30]
    print nltk.metrics.scores.accuracy(predicted,new_sentiment_collection)
    print confusion_matrix(new_sentiment_collection,predicted,labels=["positive","negative","neutral"])