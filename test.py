from itertools import combinations

import pandas as pd

from cluster import FuzzyKmodes


df = pd.read_csv('Data_fuzzy-shrooms.csv').dropna()
print(df.shape)
X = df.drop('class', axis=1)
print(X.columns)
columns = combinations(X.columns, 2)
y = df['class']
# nancols = ['bruises', 'veil-type']
for i in range(1, 10):
    m = 1 + (i * 0.1)
    clust = FuzzyKmodes(X, y, m=m)
    clust.cluster()
    print(m)
    print(clust.count_values(clust.cluster_membership()))
'''
for col in columns:
    clust = FuzzyKmodes(X[list(col)].dropna(), y)
    print(col)
    clust.cluster()
    print(clust.count_values(clust.cluster_membership()))
'''
