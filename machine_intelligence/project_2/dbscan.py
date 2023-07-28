# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:50:11 2020

@author: Manish
"""

#import things
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

### load data files
testing_attack = np.load("./data/testing_attack.npy")
testing_normal = np.load("./data/testing_normal.npy")
training_normal = np.load("./data/training_normal.npy")

### transform data files
pca = PCA(n_components=2)
training_normal_reduced = pca.fit_transform(training_normal)

### plot testing dataset
#plt.scatter(training_normal_reduced[:,0],training_normal_reduced[:,1])

#plt.scatter(np.arange(len(training_normal_reduced)),training_normal_reduced[:,1])


### Minkowski distance: p=1 Manhattan, p=2 Euclidean
dist= lambda x, y, p: np.sum(abs(x - y)**p, axis=1)**(1/p)

### calculate distance between points
dist_sheet = np.array([dist(x,training_normal_reduced,2) for x in training_normal_reduced])

### custom DBSCAN
def dbscan(eps, min_samples, dist_sheet):
    neighbours_sheet = np.less(dist_sheet,eps)
    core =  np.greater_equal(neighbours_sheet.sum(axis=0) - 1, min_samples)
    # labels = [0] * len(neighbours_sheet)
    # L = 0
    # for i in range(len(neighbours_sheet)):
    #     print(i)
    #     if labels[i] != 0:
    #         continue
    #     L += 1
    #     neighbours = np.where(neighbours_sheet[:,i])[0]
    #     if len(neighbours) < min_samples:
    #         labels[i] = -1
    #     else:
    #         neighbours = np.delete(neighbours,i)
    #         labels[i] = L
    #         j = 0
    #         while j < len(neighbours):
    #             index = neighbours[j]
    #             if labels[index] == -1:
    #                 labels[index] = L
    #             elif labels[index] == 0:
    #                 labels[index] = L
    #                 temp_neighbours = np.where(neighbours_sheet[:,index])[0]
    #                 if len(temp_neighbours) >= min_samples:
    #                     neighbours = np.append(neighbours,temp_neighbours)
    #             j += 1
    return core

### Hyper parameters range
eps_range = [0.05, 0.1, 0.5]
min_samples_range = [3, 5, 10]
### Hyper parameters combinations
parameters = [(e,m) for e in eps_range for m in min_samples_range]
#### output dataframe
df = pd.DataFrame(columns=['eps','min_samples','TP',
                           'FP', 'TN', 'FN', 'Accuracy',
                           'DR', 'FAR'])

for i, (eps, min_samples) in enumerate(parameters):
    print(i)
    df.loc[i,'eps'] = eps
    df.loc[i,'min_samples'] = min_samples
    core = dbscan(eps, min_samples, dist_sheet)
    ### Check if a point in testing_normal is normal
    dist_sheet_test_normal = np.array([dist(x,training_normal[core],2) for x in testing_normal])
    neighbours_sheet_test_normal = np.less(dist_sheet_test_normal, eps)
    
    normal =  np.greater_equal(neighbours_sheet_test_normal.sum(axis=1) - 1, 1)
    
    df.loc[i,'TN'] = normal.sum()
    df.loc[i,'FP'] = (normal==False).sum()
    
    ### Check if a poit in testing_attack is anamolous
    dist_sheet_test_attack = np.array([dist(x,training_normal[core],2) for x in testing_attack])
    neighbours_sheet_test_attack = np.less(dist_sheet_test_attack, eps)
    
    malicious =  np.less(neighbours_sheet_test_attack.sum(axis=1) - 1, 1) 
    
    df.loc[i,'TP'] = malicious.sum()
    df.loc[i,'FN'] = (malicious==False).sum()
    

df['Accuracy'] = (df['TP'] + df['TN'])/(df['TP'] + df['TN'] + df['FP'] + df['FN'])
df['DR'] = df['TP']/(df['TP'] + df['FN']) ### detection rate
df['FAR'] = df['FP']/(df['TN'] + df['FP']) ### False Alarm rate

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
