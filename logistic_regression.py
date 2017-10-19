# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:14:46 2017

@author: DIU
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

knn = KNeighborsClassifier()

lr = LogisticRegression()


iris = load_iris()
x = iris.data
y = iris.target

lr.fit(x, y)
knn.fit(x, y)
p = lr.predict([[4, 4.5, 5, 6]])
q = knn.predict([[4, 4.5, 5, 6]])
print(p)
print(q)