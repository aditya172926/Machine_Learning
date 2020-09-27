import numpy
import pandas
import datetime
import pandas_datareader.data as web
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from pandas import Series, DataFrame

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pandas.set_option('display.max_columns', 10)


start = datetime.datetime(2010, 4, 20)
end = datetime.datetime(2020, 5, 2)

df = web.DataReader("AMZN", 'yahoo', start, end)
print(df.head())
print(df.tail())

#Rolling mean and Return rate of stocks
'''Rolling mean/moving average smooths out the price data by creating a constantly updated average price
This is usefult o cut down the noise in out price chart. This moving average could act as a resistance
meaning from the downtrend and uptrend of stocks you could expect it will follow the trend and less 
likely to deviate outside the resistance point'''
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean() 
'''This will calculate the Moving average for the last 100 Windows
(100 days) and take average of each windows moving average'''
print(mavg.tail(10))

style.use('ggplot')
close_px.plot(label='AMZN')
mavg.plot(label = 'mavg')
plt.legend()
plt.show()

#Return Deviation to determine risk and return
'''Expected mean measures the mean, or expected value of the probability distribution of investments
returns. The expected return of a portfolio is calculated by multiplying the weight of each asset 
by its expected return and adding the values for each investment. '''

#plotting the return rate
rets = close_px/close_px.shift(1)-1
rets.plot()
plt.show()

'''Logically the returns must be as high and stable as possible.'''

#Analysing Competitor stocks
dfcomp = web.DataReader(['AMZN', 'GE', 'GOOG', 'IBM'], 'yahoo', start = start, end = end)['Adj Close']
print(dfcomp.head())
print(dfcomp.tail())

#Correlation Analysis
'''We can analyse the competition by running the percentage change and correlation functions in pandas.
Percentage changes will find out how much the price change compared to the previous day which defines
returns. Knowing the correlation will help us to find out whether the returns are affected by the other
stock returns.'''
competitor_return = dfcomp.pct_change()
corr = competitor_return.corr()
print('\n')
#print(corr)

'''Lets plot apple and GE with scatter plot to view their return distribution.
It will give you a visual of how stock returns of one company affects the other'''
plt.scatter(competitor_return.AMZN, competitor_return.GE)
plt.xlabel('Amazon returns')
plt.ylabel('GE returns')
plt.show()



#a = competitor_return['AAPL'].values
#print(len(a))


#Predicting Stock Price ---------------------------------------------------------------------
# #Feature Engineering
# dfreg = df.loc[:, ['Adj Close', 'Volume']]      #all the rows of these 2 columns
# dfreg['HL_PCT'] = ((df['High']-df['Low'])/df['Close']) * 100.0
# dfreg['PCT_change'] = ((df['Close']-df['Open'])/df['Open']) * 100.0
# print(dfreg.head())

# #Cleaning the data

# #Drop missing values
# dfreg.fillna(value = -99999, inplace = True)

# #Separate 1% of the data to forecast
# forecast_out = int(math.ceil(0.01 * len(dfreg)))

# #Separating the label here, we want to predict the Adj Close
# forecast_col = 'Adj Close'
# dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
# print(dfreg.columns)

# X = numpy.array(dfreg.drop(['label'], 1))

# #Scale X so that everyone can have the same distribution for the linear regression
# X = preprocessing.scale(X)

# #Find datasets of late X and early X (train) for model generation and evaluation
# X_lately = X[-forecast_out:]
# X = X[:-forecast_out]

# #Separate label and identify it as y
# y = numpy.array(dfreg['label'])
# y = y[:-forecast_out]

# print(X.shape)
# print(y.shape)

# #Separation of training and testing of model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# #Model generation

# '''Simple Linear analysis shows the linear relationship between two or more variables.
# Quadratic Discriminant analysis is similar to Linear analysis, except that it allows polynomial and displays curves'''


# #Linear regression
# clfreg = LinearRegression(n_jobs=-1)
# clfreg.fit(X_train, y_train)

# confidence_l = clfreg.score(X_test, y_test)
# print(confidence_l)

# #Lasso
# clfl = Lasso()
# clfl.fit(X_train, y_train)

# confidence_lasso = clfl.score(X_test, y_test)
# print(confidence_lasso)

# #Ridge
# clfr = Ridge()
# clfr.fit(X_train, y_train)

# confidence_r = clfr.score(X_test, y_test)
# print(confidence_r)

# #KNN Regression
# clfknn = KNeighborsRegressor(n_neighbors=2)
# clfknn.fit(X_train, y_train)

# confidence_knn = clfknn.score(X_test, y_test)
# print(confidence_knn)

# def plotPrediction (_forecast_set, _predictionName, _lastDate):
#     dfreg['Forecast'] = numpy.nan
#     last_unix = _lastDate
#     next_unix = last_unix + datetime.timedelta(days=1)

#     for i in _forecast_set:
#         next_date = next_unix
#         next_unix += datetime.timedelta(days=1)
#         dfreg.loc[next_date] = [numpy.nan for _ in range(len(dfreg.columns)-1)]+[i]

#     mpl.rc('figure', figsize=(8,7))
#     style.use('ggplot')

#     dfreg['Adj Close'].tail(300).plot()
#     dfreg['Forecast'].tail(300).plot()

#     plt.title(_predictionName)
#     plt.legend(loc=4)
#     plt.xlabel('Date')
#     plt.ylabel('Price')

#     plt.show()


# last_date = dfreg.iloc[-1].name

# forecast_set = clfreg.predict(X_lately)
# print('Linear regression prediction:')
# print(forecast_set)
# plotPrediction(forecast_set, 'Linear Regression Prediction', last_date)
# print('\n')
# forecast_set = clfknn.predict(X_lately)
# print('KNN prediction:')
# print(forecast_set)
# plotPrediction(forecast_set, 'KNN', last_date)
# print('\n')
# forecast_set = clfl.predict(X_lately)
# print('Lasso prediction:')
# print(forecast_set)
# plotPrediction(forecast_set, 'Lasso', last_date)
# print('\n')
# forecast_set = clfr.predict(X_lately)
# print('Ridge prediction:')
# print(forecast_set)
# plotPrediction(forecast_set, 'Ridge', last_date)