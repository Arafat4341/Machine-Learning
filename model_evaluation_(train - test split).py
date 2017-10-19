# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:52:11 2017

@author: DIU
"""

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics as mt

iris = load_iris()
x = iris.data
y = iris.target

lr = LogisticRegression()
knn = KNeighborsClassifier()
knn2 = KNeighborsClassifier(n_neighbors = 1)

#percentage of test values.  0.4 means 40%
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 4)

lr.fit(x_train, y_train)
knn.fit(x_train, y_train)
knn2.fit(x_train, y_train)

lr_predict = lr.predict(x_test)
kp = knn.predict(x_test)
kp2 = knn2.predict(x_test)

accu = mt.accuracy_score(lr_predict, y_test)
kaccu = mt.accuracy_score(kp, y_test)
kaccu2 = mt.accuracy_score(kp2, y_test)

print('logistic accuracy ', accu)
print('KNN accuracy: ', kaccu)
print('KNN accuracy with k = 1: ', kaccu2)


