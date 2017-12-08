import numpy as np
import pandas as pd
import random
from collections import Counter


df = pd.read_csv('Data_fuzzy-shrooms.csv')
# X = pd.read_csv('https://raw.githubusercontent.com/medhini/Genetic-Algorithm-Fuzzy-K-Modes/master/zoo.csv')
y = df['class'].values
X = df
k = 5
m = 1.2
n_iter = 10
p = len(X.columns)
n = len(X)
columns = X.columns

def random_numbers(x, attribute):
    xs = x[attribute].unique()
    randoms = np.random.dirichlet(np.ones(len(xs)), size=1)
    return list(zip(xs, randoms[0]))

def init(x, k):
    return [[random_numbers(x, attribute)
            for attribute in columns]
            for _ in range(k)]

def dissimilarity(fuzzy_set, x_l):
    return sum(0 if value == x_l else confidence
               for value, confidence in fuzzy_set)

def distance(cluster, x):
    return sum(dissimilarity(fuzzy_set, x[j])
               for j, fuzzy_set in enumerate(cluster))

def membership(x, cluster, clusters):
    exp = 1 / ( m - 1 )
    membership = sum((distance(cluster, x) / distance(other_cluster, x))** exp
               # if cluster != other_cluster else 1
               for other_cluster in clusters)
    return membership ** -1

def gamma(x_j, attribute, membership):
    if x_j == attribute:
        return membership ** m
    else:
        return 0

def certainty(Xj, value, u_row):
    new_memb = sum(u_row[i]**m if x == value else 0 for i, x in enumerate(Xj))
    return new_memb

def update_clusters(clusters, X, u):
    # Update the clusters
    for j, cluster in enumerate(clusters):
        cl_u = u[j]
        for l, attribute_membership in enumerate(cluster):
            feature = columns[l]
            clusters[j][l] = [(value, certainty(X[feature].values, value, cl_u))
                              for value, _ in attribute_membership]
    return clusters

def check_normal(clusters, iter):
    clusters = [[normalize(row) for row in cluster]
                for cluster in clusters]
    for cluster in clusters:
        for row in cluster:
            if not (0.9 < sum(r[1] for r in row) < 1.1):
                print('error', row)
                print('error', sum(r[1] for r in row))
                raise Exception(str(iter) + str(cluster))
    return clusters

def normalize(row):
    summed = sum(r[1] for r in row)
    return [(value, x/summed) for value, x in row]

def main():
    clusters = init(X, k) # Step 1
    clusters = check_normal(clusters, -2)
    u = [[membership(x, cluster, clusters) for x in X.values] # Step 2
        for cluster in clusters]
    clusters = update_clusters(clusters, X, u) # Step 3
    clusters = check_normal(clusters, -1)

    for _ in range(n_iter):
        new_u = [[membership(x, cluster, clusters) for x in X.values] # Step 2
                for cluster in clusters]
        clusters_new = update_clusters(clusters, X, new_u) # Step 3
        clusters_new = check_normal(clusters_new, _)
        u_error = np.mean([abs(m - new_m) for i in range(k)
                    for m, new_m in zip(u[i], new_u[i])])
        print(_)
        print('error', u_error)
        # print(clusters)
        u = new_u
        clusters = clusters_new

    cluster_membership = {i: list() for i in range(k)}
    for i in range(n):
        cluster = max((u[cl][i], cl) for cl in range(k))[-1]
        cluster_membership[cluster].append(y[i])

    for cl, vals in cluster_membership.items():
        print(cl, Counter(vals))

if __name__ == '__main__':
    main()
