# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:04:29 2020

@author: Manish

Modified on Fri Sep 11, 11:27 2020
"""

# import os
import string
import numpy as np
import pandas as pd
import nltk
import time
import concurrent.futures

### Set working 
# os.chdir('/home/ms8515/CA_ML/')

### Download stopwords and punctuations
nltk.download('stopwords')
nltk.download('punkt')
punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))

### Load SMSSpamCollection file
# with open('SMSSpamCollection') as f:
#     lines = f.readlines()

### Load SMSSpamCollection file using numpy
#file = np.loadtxt('SMSSpamCollection')

### Load SMSSpamCollection file using pandas
file = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label','msg'])

### lowercase
file['msg'] = file['msg'].apply(lambda x: x.lower())

### remove punctuations
file['msg'] = file['msg'].apply(lambda x: ''.join([word for word in x if word not in punctuations]))

### remove stopwords and replace with space
# file['msg'] = file['msg'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))

### drop empty rows
file = file[~file['msg'].isin([''])]

### tokenize
file['msg'] = file['msg'].str.split(' ')


### generate random index mask for a fix seed 
np.random.seed(seed=111)
arr = np.arange(len(file))
np.random.shuffle(arr)

### custom train-test split (80:20)
train = file.iloc[arr[:int(len(arr)*0.8)],:].reset_index(drop=True)
test = file.iloc[arr[int(len(arr)*0.8):],:].reset_index(drop=True)

### Unique words in train set messages
train_unique = []
for x in train['msg']:
    train_unique.extend(x)
train_unique = np.unique(train_unique)
del x, arr, punctuations, stopwords, file

### train_idf
train_idf = np.array([np.log(len(train)/len([x for y in train['msg'] if x in y])) for x in train_unique])

### train_tf and test_tf
train_tf = np.array([[y.count(x)/len(y) for y in train['msg']] for x in train_unique]).transpose()
test_tf = np.array([[y.count(x)/len(y) for y in test['msg']] for x in train_unique]).transpose()

### train and test tf_idf
train_tf_idf = train_tf * train_idf
test_tf_idf = test_tf * train_idf
del train_tf, test_tf, train_idf, train_unique

### train and test labels: spam: 1 and ham: 0
train_label = np.array([1 if x=='spam' else 0 for x in train['label']])
test_label = np.array([1 if x=='spam' else 0 for x in test['label']])
del train, test

### Minkowski distance: p=1 Manhattan, p=2 Euclidean
dist= lambda x, y, p: np.sum(abs(x - y)**p, axis=1)**(1/p)

### Custom K-Nearest Neighbours Model: Classes
knn = lambda a, a_l, b, k, p: a_l[np.argsort(dist(a, b, p))[:k]]

### Determine most frequent occuring class
predict = lambda a, a_l, b, k, p: np.bincount(knn(a, a_l, b, k, p)).argmax()

### calculate stats
def stats(test_y, pred):
    ### True Positive
    TP = np.where(test_y==1, pred==test_y, 0).sum()
    ### False Positive
    FP = np.where(test_y==0, pred!=test_y, 0).sum()
    ### False Negative
    FN = np.where(test_y==1, pred!=test_y, 0).sum()
    ### True Negative
    TN = np.where(test_y==0, pred==test_y, 0).sum()
    ### accuracy
    # A = np.round(((test_y==pred) + 0).sum()/len(test_y) * 100, 3)
    A = (TP+TN)/(TP+FP+TN+FN)
    ### Precision
    P = TP/(TP+FP)
    ### Recall
    R = TP/(TP+FN)
    return A, P, R

### main function
def fun(train_x, train_y, test_x, test_y, k, p):
    print("Started thread:  K: " + str(k)+ "  P: "+ str(p))
    start_time = time.time()
    ### prediction
    pred = np.array([predict(train_x, train_y, x, k, p) for x in test_x])
    ### stats: Accuracy, Precision and Recall
    A, P, R = stats(test_y, pred)
    print("Stopped thread:  K: " + str(k) + "  P: "+ str(p))
    return {'K': k, 'Minkowski (p)': p, 'Accuracy': A, 'Precision': P, 
            'Recall': R, 'time (s)': time.time() - start_time}


#this section starts up a variety of threads through concurrent futures 
#to loop through p values of 1,2,3 and k values of 2,5,8,10.
#results are then stored to be converted to a DataFrame.
threads = []
thread_outputs = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    for p in [1,2,3]:
        for k in [2,5,8,10]:
            #print("starting thread: p: " + str(p) + " k: " + str(k)) 
            future = executor.submit(fun, train_tf_idf, train_label, test_tf_idf, test_label, k, p) 
            threads.append(future)
    for i in threads:
        thread_outputs.append(i.result())
output = pd.DataFrame.from_dict(thread_outputs)
#output = pd.DataFrame([fun(train_tf_idf, train_label, test_tf_idf, test_label, k, p) for p in [1] for k in [2]])
### Evaluation over k={2,5,8,10} and Minkowski Coefficient={1,2,3}
# output = pd.DataFrame([fun(train_tf_idf, train_label, test_tf_idf, test_label, k, p) for p in [1,2,3] for k in [2,5,8,10]])
output.to_excel('output.xlsx', index=False)


###############################################################################
### with scaling [0,1]

# x_min, x_max = np.min(train_tf_idf, axis=0), np.max(train_tf_idf, axis=0)
# train_tf_idf_scaled = (train_tf_idf - x_min)/(x_max - x_min)
# test_tf_idf_scaled = (test_tf_idf - x_min)/(x_max - x_min)

# output = pd.DataFrame([fun(train_tf_idf_scaled, train_label, test_tf_idf_scaled, test_label, k, p) for p in [1,2,3] for k in [2,5,8,10]])
# output.to_excel('output_scaled.xlsx', index=False)


###############################################################################
### Cosine Similarity as a distance metric
# dist = lambda x, y: np.dot(x,y)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y))

# ### Custom K-Nearest Neighbours Model: Classes
# knn = lambda a, a_l, b, k: a_l[np.argsort(dist(a, b))[:k]]

# ### Determine most frequent occuring class
# predict = lambda a, a_l, b, k: np.bincount(knn(a, a_l, b, k)).argmax()

# ### main function
# def fun(train_x, train_y, test_x, test_y, k):
#     start_time = time.time()
#     ### prediction
#     pred = np.array([predict(train_x, train_y, x, k) for x in test_x])
#     ### stats: Accuracy, Precision and Recall
#     A, P, R = stats(test_y, pred)
#     return {'K': k, 'Accuracy': A, 'Precision': P, 'Recall': R, 
#             'time (s)': time.time() - start_time}

# output = pd.DataFrame([fun(train_tf_idf, train_label, test_tf_idf, test_label, k) for k in [2,5,8,10]])
# output.to_excel('output_cosine_similarity.xlsx', index=False)