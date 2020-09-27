import numpy
import pandas
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize']=(20.0, 10.0)

os.chdir('/Users/harshshetye/Desktop/Machine Learning/Datasets')
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

data = pandas.read_csv('iris (1).csv', names=names)
print(data.shape)
#print(data.head())

#we can get the linear relationship between sepal-length and sepal-width
#collecting x and y values
x = data['sepal-length'].values
y = data['sepal-width'].values

#calculate the mean of x and y values
mean_x = numpy.mean(x)
mean_y = numpy.mean(y)

#total number of values
n = len(x)

#Using the formula to calculate the value of m and c
numer = 0
denom = 0
for i in range(n):
    numer += (x[i]-mean_x)*(y[i]-mean_y)
    denom += (x[i]-mean_x)**2
m = numer/denom
c = mean_y - (m*mean_x)

#print m and c... m is the slope and c is the constant
print(m,c)

#PLotting=====================================
max_x = numpy.max(x)+4
min_x = numpy.min(x)-4
X = numpy.linspace(min_x, max_x, 10)
Y = c + m*X

#line plotting
plt.plot(X, Y, color='Red', label='Regression line')
#scatter plot
plt.scatter(x, y, c='Green', label='Scatter plot')

plt.xlabel('Sepal-length')
plt.ylabel('Sepal-width')
plt.legend()
plt.show()

#TO see how good our model is by calculating R square
ss_t = 0
ss_r = 0
for i in range(n):
    y_pre = c + (m*x[i])
    ss_t += (y[i]-mean_y)**2
    ss_r += (y[i]-y_pre)**2
R2 = 1 - (ss_r/ss_t)
print(R2)


