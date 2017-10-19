# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:21:32 2017

@author: DIU
"""

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=5)
score = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
print(score)
print(score.mean())

