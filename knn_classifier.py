# CC BY-NC-SA
# Copyright 2022. Suhwan Lim. all rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

import tensorflow as tf
import csv
import random
import math
import operator
import cv2
import numpy as np
from color_histogram_feature_extraction import color_util

# calculation of euclidead distance
def calculateEuclideanDistance(variable1, variable2):
    distance = 0
    distance=np.sum(np.absolute(variable1-variable2[:, 0:3].astype("int8")), axis=1)
    #distance=np.sqrt(distance)
    return np.reshape(distance, (len(distance),1)) 

# get k nearest neigbors
def kNearestNeighbors(training_feature_vector, testInstance, k):
    length = len(testInstance)
    dist=calculateEuclideanDistance(testInstance, training_feature_vector)
    distances=np.hstack((training_feature_vector, dist))
    mask=np.argsort(distances[:, 4], axis=0)
    distances=distances[mask.T]
    return distances[0:k]

# votes of neighbors
def responseOfNeighbors(neighbors):
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][3]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sortedVotes = sorted(all_possible_neighbors.items(),
                        key=operator.itemgetter(1), reverse=True) 
        
    return sortedVotes[0][0]


# Load image feature data to training feature vectors and test feature vector
def loadDataset(img, 
    label,
    filename
    ):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        train_data=np.array(dataset)

    dataset=color_util(img, label)
    dataset, label=dataset.color_histogram_of_test_image()
    try:
        test_data=np.array((dataset[label]))
    except KeyError:
        pass

    try:
        return train_data, test_data
    except UnboundLocalError:
        pass     

def main(training_data, img, label):
    try:
        train_data, test_data=loadDataset(img, label, training_data)
    except TypeError or UnboundLocalError:
        pass

    classifier_prediction = []  # predictions
    k = 5  # K value of k nearest neighbor
    #for x in range(len(test_feature_vector)):
    neighbors = kNearestNeighbors(train_data, test_data, k)
    result = responseOfNeighbors(neighbors)
    classifier_prediction.append(result)
    classifier_prediction=tuple(classifier_prediction)
    return classifier_prediction[0]		
