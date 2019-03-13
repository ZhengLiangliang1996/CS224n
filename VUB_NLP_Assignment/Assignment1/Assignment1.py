#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:09:10 2019

@author: liangliang
"""

# module loading 
import gzip
import numpy as np
import pandas as pd

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Musical_Instruments_5.json.gz')


#%%

import nltk
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# word preprocessing for baseline
def preprocessing_baseline(review):
    '''
        remove punctuation
    '''
    result_review = []
    for sub_review in review:
        
        words = nltk.word_tokenize(sub_review)

        words=[word.lower() for word in words if word.isalpha()]
        result_review.append(words)
    return result_review


def build_models(X_train, y_train, X_test, y_test, feature_range,feature_step):
    model_list = []
    # feature means coloum in the dataset
    for features in range(feature_range[0],feature_range[1],feature_step):
        tfidF = TfidfVectorizer(max_features=features)
        
        # transfer to tfidf
        train_set = tfidF.fit_transform(x_train)
        test_set = tfidF.transform(x_test)   

        # Naive Bayes
        model = MultinomialNB()
        model.fit(train_set, y_train)
        
        # dict 
        r = {}
        r['features'] = features
        r['train_acc'], r['test_acc'], r['train_f1'], r['test_f1'], r['tr_cf'] , r['te_cf'], _, _ = get_train_test_score(model,
                                                                                                                         train_set, 
                                                                                                                         test_set, 
                                                                                                                         y_train,                                                                                                                       y_test)
        model_list.append(r)
        return model_list
    

def get_train_test_score(model, x_train, x_test, y_train, y_test):
    
    """
    Function to get train and test score
    """
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_acc = accuracy_score(y_train,y_train_pred)
    test_acc = accuracy_score(y_test,y_test_pred)
    
    train_f1 = f1_score(y_train,y_train_pred,average='weighted')
    test_f1 = f1_score(y_test,y_test_pred,average='weighted')
    
    train_cf = confusion_matrix(y_train,y_train_pred)
    test_cf = confusion_matrix(y_test,y_test_pred)
    
    train_acc, test_acc, train_f1, test_f1 = [round(x*100,1) for x in [train_acc, test_acc, train_f1, test_f1]]
    
    return train_acc, test_acc, train_f1, test_f1, train_cf, test_cf, y_train_pred, y_test_pred




pp_types = ['BASE']

pp_desc = {  'BASE': 'Baseline'
          }

models = dict((pp,[]) for pp in pp_types)    



# getting data 

def scorePreprocessor(score):
    # 1  -> positive 
    # -1 -> negative
    # 0  -> neutral 
    res = []
    for i in range(len(score)):
        if score[i] >= 3.0:
            res.append(1)
        else:
            res.append(0)
    return res

x_train, x_test, y_train, y_test = train_test_split(df.reviewText[0:2],
                                                    scorePreprocessor(df.overall[0:2]),
                                                    stratify=scorePreprocessor(df.overall[0:2]),
                                                    test_size=0.2,
                                                    random_state=20)


print(len(x_train),len(x_train),len(x_test),len(y_test))

x_train_b = preprocessing_baseline(x_train)

x_test_b = preprocessing_baseline(x_test)

feature_range = (15000, 45000)

models['BASE'] = build_models(x_train_b,y_train,x_test_b,y_test,feature_range, 500)

plt.rcParams['figure.figsize'] = [18,4]

for pp in pp_types:
    plot_acc_f1(pp,feature_range)