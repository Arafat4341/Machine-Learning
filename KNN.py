# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:41:02 2017

@author: DIU
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data

y = iris.target

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x, y)

ak = knn.predict([[4, 5, 6, 2], [2, 3, 4.4, 8]])

print(ak)