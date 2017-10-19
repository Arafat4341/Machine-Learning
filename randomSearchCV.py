# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:40:23 2017

@author: DIU
"""

from sklearn.grid_search import RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


k_range =[]
for i in range(31):
    k_range.append(i+1)

w = ['uniform','distance']
    
param_dist = dict(n_neighbors=k_range, weights=w)

iris = load_iris()
x = iris.data
y = iris.target

knn = KNeighborsClassifier()
rnd = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy',
                          n_iter = 10, random_state = 5)

rnd.fit(x,y)
print(rnd.grid_scores_)