#!/bin/python3
# import things
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

from random import randint

# import sklearn/scikit stuff
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# define global variables for training/testing data
testing_attack = []
testing_normal = []
training = []

threshold = 0.1

# this method generates the average for an array of arrays
# or in this case, an array of training matrices.
def array_average(arrays):
    return np.average(np.array(arrays), axis=0)
    chonk_array = [arrays[0]]

# This method is no longer used, but I wanted to make sure that it got here so you could 
# at least see the logic we used to determine k.
def determine_k(classification, tolerance, iterations, training):
    centroids = []
    for i in range(3,20):
        data = kmeans_train(i, tolerance, iterations, training)
        centroids.append(data)
    values_in_cluster_average = []
    for i in range(3,20):
        data = centroids[i-3][0]
        biggest_cluster = 0
        for z in data.keys():
            if(len(data[z]) > biggest_cluster):
                biggest_cluster = len(data[z])
        print("K=" + str(i) + ": " + str(biggest_cluster))

def kmeans_train(k, tolerance, iterations, dataset):
    # Create empty centroids dictionary
    centroids = []
    usedcents = []

    # Choose the first k points to be the centroids
    for x in range(k):
        # randomly determine a centroid
        randomcent = randint(0, len(dataset))
        # ensure the centroid isn't already in use
        if randomcent in usedcents:
            x -= 1
        else:
            centroids.append(dataset[randomcent])
        usedcents.append(randomcent)
    for i in range(0, iterations):
        classification = kmeans_classify(dataset, centroids)
        # calculate new centroids
        for i in range(0, k):
            # this calculates the centroid, by first calculating the array average for
            # every piece of raw data in the training array.
            centroids[i] = array_average([(dataset[n]) for n in classification.get(str(i))])
    return (centroids)


# classifies data.  Expects an array of an array of samples along with the centroids generated
# by the kmeans_train method.
def kmeans_classify(dataset, centroids):
    ## generate euclidean distances for training data based on the centroids
    classification = {}
    distances = euclidean_distances(dataset, centroids)
    # Iterate through each point and determine which centroid it belongs to
    for point in range(len(dataset)):
        # Iterate through each distance value for the point to determine minimum value
        points = np.where(distances[point] == np.min(distances[point]))
        # Above returns the index of the centroid. Assign the point to that centroid.
        # points[0] is the index

        # the below generates a variable for the position of the lesser variable in the dictionary
        centroidindex = points[0][0]
        centroidStr = str(centroidindex)
        if (centroidStr in classification):
            classification[centroidStr].append(point)
        else:
            classification[centroidStr] = []
            classification[centroidStr].append(point)
    return (classification)

def k_means_determine(dataset,centroids):
    ## generate euclidean distances for training data based on the centroids
    distancetocent = []
    dictionary = {}
    counts = {}
    classification = []
    distances = euclidean_distances(dataset, centroids)
    # Iterate through each point and determine which centroid it belongs to
    for point in range(len(dataset)):
        # Iterate through each distance value for the point to determine minimum value
        points = np.where(distances[point] == np.min(distances[point]))
        # Above returns the index of the centroid. Assign the point to that centroid.
        # points[0] is the index

        # the below generates a variable for the position of the lesser variable in the dictionary
        centroidindex = points[0][0]
        centroidStr = str(centroidindex)

        distancetocent.append(distances[point][centroidindex])

        if (centroidStr in dictionary):
            dictionary[centroidStr] = dictionary[centroidStr] + distances[point][centroidindex]
            counts[centroidStr] = counts[centroidStr] + 1
        else:
            dictionary[centroidStr] = distances[point][centroidindex]
            counts[centroidStr] = 1
    for key in dictionary:
        dictionary[key] = dictionary[key] / counts[key]

    for point in range(len(dataset)):
        # Iterate through each distance value for the point to determine minimum value
        points = np.where(distances[point] == np.min(distances[point]))
        # Above returns the index of the centroid. Assign the point to that centroid.
        # points[0] is the index

        # the below generates a variable for the position of the lesser variable in the dictionary
        centroidindex = points[0][0]
        centroidStr = str(centroidindex)

        if (distancetocent[point] < threshold):
            classification.append('0')
        else:
            classification.append('1')
    return (classification)

def main():
    # load in data files
    testing_attack = np.load("./data/testing_attack.npy")
    testing_normal = np.load("./data/testing_normal.npy")
    training = np.load("./data/training_normal.npy")

    # set the basic arguments
    k = 4
    tolerance = 0.001
    iterations = 150

    # Testing array, ignore
    x = np.array([[1, 2],
                  [1.5, 1.8],
                  [5, 8],
                  [8, 8],
                  [1, 0.6],
                  [9, 11]])

    # run scikit-learn's PCA algorithm
    pca = PCA(n_components=2)
    training_data = pca.fit_transform(training)

    # Testing prints
    # print(training[0])
    # distances = euclidean_distances(training[0],training[1])
    # print(training_data)
    # print(distances)

    # Data plot sections
    subdata = training_data
    subdata2 = training[0:10000]
    subdata3 = training[0:100]

    # distances = euclidean_distances(subdata2,subdata3)

    # print(distances)

    x_new = pca.inverse_transform(training_data)
    plt.scatter(training[:, 0], training[:, 1], alpha=0.2)
    plt.scatter(x_new[:, 0], x_new[:, 1], alpha=0.8)
    plt.axis('equal')
    # plt.show()

    testing_full = np.concatenate((testing_normal, testing_attack))

    classification = kmeans_train(k, tolerance, iterations, training)
    outcome = k_means_determine(testing_full, classification)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(outcome)):
        if(i < len(testing_normal)):
            if(int(outcome[i]) == 0):
                tn += 1
            else:
                fp += 1
        else:
            if(int(outcome[i]) == 0):
                fn += 1
            else:
                tp += 1

    print(outcome)
    print("TP:", tp)
    print("FP:", fp)
    print("TN:", tn)
    print("FN:", fn)


main()