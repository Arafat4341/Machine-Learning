# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:28:45 2017

@author: DIU
"""

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import pandas as pd

data = pd.read_csv('files/ad.csv', index_col = 0)
print(data.head())
x = data[['TV','Radio','Newspaper']]
y = data['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)
linreg = LinearRegression()

linreg.fit(x_train, y_train)

print(linreg.coef_, linreg.intercept_)