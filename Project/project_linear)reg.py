import numpy
import pandas
import os
import matplotlib.pyplot as plt
import math

os.chdir('/Users/harshshetye/Desktop/Machine Learning/Datasets/residential_data')
data = pandas.read_csv('1.csv')
x = data['TS'].values
y = data['V1'].values
print(numpy.min(x))
print(numpy.max(x))
print(numpy.min(y))
print(numpy.max(y))
t = len(x)
n = len(x)//10      #Try to split the dataset into 10 parts

X = x  #slicing
#print(X)
#X1 = X.reshape((n,1))
#print(X1)   #The array of 7504 rows is now in 10 parts of 750 rows each
Y = y
#Y1 = Y.reshape((n,1))

X_mean = numpy.mean(X)
Y_mean = numpy.mean(Y)

#To calculate m and c
numer = 0
denom = 0

for i in range(t):
    numer += ((X[i]-X_mean)*(Y[i]-Y_mean))
    denom += ((X[i]-X_mean)**2)
m = numer/denom
print(m)
c = Y_mean - (X_mean*m)
print(c)
#PLotting
max_X = numpy.max(X)+10
min_X = numpy.min(X)-10

A = numpy.linspace(min_X, max_X, 7509)
print(A)
print('\n')
B = m*A + c
print(B)

#line plotting
plt.plot(A, B, color='Red', label='Regression Line')    #can be used for prediction
#scatter plot
plt.scatter(X, Y, c='Green', label='Scatter plot')      #Actual Values

plt.xlabel('Time')
plt.ylabel('Voltage phase 1')
plt.legend()
plt.show()
