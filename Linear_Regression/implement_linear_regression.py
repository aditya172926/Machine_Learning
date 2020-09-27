#Implementation of Linear regression with machine learning scikit learn
import numpy
import pandas
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# print(os.getcwd())
# os.chdir('/Users/harshshetye/Desktop')
# print(os.getcwd())

# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# data = pandas.read_csv('iris (1).csv', names=names)

# x = data['sepal-length'].values
# y = data['sepal-width'].values
# #cannot use rank 1 matrix in scikit learn
# n = len(x)
# x = x.reshape((n,1)) #making a matrix

# #Creating model
# reg = LinearRegression()
# #Fitting trainig data
# reg = reg.fit(x, y)

# #Y prediction
# y_pre = reg.predict(x)

# #Calculating R2 score
# R2 = reg.score(x, y)

# print(R2)


rng = numpy.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2*x - 5 + rng.rand(50)

plt.scatter(x, y)
plt.show()

model = LinearRegression(fit_intercept=True)
model.fit(x[:, numpy.newaxis], y)
xfit = numpy.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, numpy.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)

plt.show()

print('Model slope : ', model.coef_)
print('Model Intercept : ', model.intercept_)

#Testing with multidimensional data

rng = numpy.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + numpy.dot(X, [1.5, -2., 1.])

model.fit(X, y)
print('Model slope : ', model.coef_)
print('Model intercept : ', model.intercept_)

