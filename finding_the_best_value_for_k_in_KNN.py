# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 23:40:06 2017

@author: DIU
"""

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

k_range = range(1,31)

k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
    k_scores.append(score.mean())
    
plt.plot(k_range, k_scores)