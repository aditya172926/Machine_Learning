import numpy
import pandas
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
print(os.getcwd())
os.chdir('/Users/harshshetye/Desktop/Machine Learning/Datasets/residential_data')

data = pandas.read_csv('1.csv')

x = data['TS'].values
y = data['V1'].values

n = len(x)
x = x.reshape((n,1))

reg = LinearRegression()
reg = reg.fit(x, y)

y_pre = reg.predict(x)

R2 = reg.score(x,y)
print(R2)
