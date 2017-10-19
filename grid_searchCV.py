# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:21:20 2017

@author: DIU
"""

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x = iris.data
y = iris.target

k_range =[]
for i in range(31):
    k_range.append(i+1)

param_grid = dict(n_neighbors=k_range)

knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

grid.fit(x, y)
print(grid.grid_scores_)
print(grid.best_score_)
print(grid.best_params_)

knn2 = KNeighborsClassifier(n_neighbors = grid.best_params_['n_neighbors'])
knn2.fit(x,y)
print(knn2.predict(x))

