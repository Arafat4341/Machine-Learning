# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:01:27 2017

@author: DIU
"""

from sklearn import tree
x = [[36, 24, 36], [36, 30, 39], [37, 30, 31], [33, 33, 34]]
y = ['female', 'female', 'male', 'male']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
pre = clf.predict([[39, 34, 35]])
print(pre)