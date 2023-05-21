# KMeans_with_Python_Functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Making a data for clustering
from sklearn.datasets import make_blobs
X = make_blobs(n_samples=100, n_features=2, 
                            centers=None, cluster_std=1.0, center_box=(-10.0, 10.0), 
                            shuffle=True, random_state=None, return_centers=False)
X = X[0]

#The function randomly get the X of K points in data and set it as primary centroids
def init_centroids_producer(X, K):
    m, n = X.shape
    randidx = np.random.permutation(m)
    init_centroids = X[randidx[:K]]
    return init_centroids

#The function assigns data points to the nearest centroid and returns the idx as the centroid id that each data point was assigned to
def cluster_assigner(X, centroids):
    m, n = X.shape
    idx = np.zeros(m)
    K = centroids.shape[0]
    for item in range(m):
        j_arr = np.zeros(K)
        for item_2 in range(K):
            j_arr[item_2] = np.sum ((X[item] - centroids[item_2])**2)
        idx[item] = np.argmin(j_arr)
    return idx

#The function compute new centroids based on the centroid assignment performed in the previous function.
def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros(n * K).reshape(K, -1)
    for item in range(K):
        a = 0
        for item_2 in range(m):
            if idx[item_2] == item:
                centroids[item] = centroids[item] + X[item_2]
                a = a + 1
            else:
                continue
        centroids[item] = centroids[item]/a
    return centroids        

#The function computes the inertia of the algorithm by summing the distance of each datapoint from centroids
def compute_inertia(X, centroids, idx):
    m, n = X.shape
    inertia = 0
    for item in range(m):
        clus = idx[item]
        inertia = inertia + np.sum((X[item] - centroids[int(clus)])**2)
    return inertia

#The function performs KMeans by iterating by the number of internal_iteration assigning data points to centroids and recomputing new centroids.
#The algorithm is repeated alg_iteration times (to minimize the effect of the primary centroids taken on the final result) and final centroids and inertia computed in each iteration are saved in arrays.
#Finally, the best inertia is determined and the centroids and inertia and idx pertinent to the best round of algorithm are returned
def KMeans(X, K, alg_iteration, internal_iteration):
    m, n = X.shape
    centroid_array = np.zeros((alg_iteration, K, n))
    inertia_array = np.zeros(alg_iteration)
    for item in range(alg_iteration):
        init_centroids = init_centroids_producer(X, K)
        centroids = init_centroids
        for item_2 in range(internal_iteration):
            idx = cluster_assigner(X, centroids)
            centroids = compute_centroids(X, idx, K)
        inertia_array[item] = compute_inertia(X, centroids, idx)
        centroid_array[item] = centroids
    best = np.argmin(inertia_array)
    idx = cluster_assigner(X, centroid_array[best])
    return centroid_array[best], inertia_array[best], idx

centroid, inertia, idx = KMeans(X, 3, 10, 20)
