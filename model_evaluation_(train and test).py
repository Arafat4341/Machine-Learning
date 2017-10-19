# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:57:26 2017

@author: DIU
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics as mt

iris = load_iris()

lr = LogisticRegression()

knn = KNeighborsClassifier(n_neighbors = 1)
knn2 = KNeighborsClassifier(n_neighbors = 5)

x = iris.data
y = iris.target

print(x,y)

lr.fit(x, y)
pre = lr.predict(x)

knn.fit(x, y)
knn2.fit(x, y)
kpre = knn.predict(x)
kpre2 = knn2.predict(x)

print(mt.accuracy_score(y, pre))

print(mt.accuracy_score(y, kpre))

print(mt.accuracy_score(y, kpre2))