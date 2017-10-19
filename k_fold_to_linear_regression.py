# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 00:12:49 2017

@author: DIU
"""

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd

data = pd.read_csv('files/ad.csv', index_col = 0)
fc = ['TV', 'Radio', 'Newspaper']
x = data[fc]
y = data['Sales']

lm = LinearRegression()
score = cross_val_score(lm, x, y,cv=10, scoring='mean_squared_error')

mse_scores = -score

rmse_scores = np.sqrt(mse_scores)

print(rmse_scores.mean())

fc2 = ['TV', 'Radio']

X = data[fc2]
Y = data['Sales']

lm2 = LinearRegression()
score2 = cross_val_score(lm2, X, Y,cv=10, scoring='mean_squared_error')

mse_scores2 = -score2

rmse_scores2 = np.sqrt(mse_scores2)

print(rmse_scores2.mean())
