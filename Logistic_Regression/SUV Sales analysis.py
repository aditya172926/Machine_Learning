#To predict if the on the basis of predicting factors who might buy the SUV
import numpy
import pandas
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

path = '/Users/harshshetye/Desktop/Machine Learning/Datasets'
os.chdir(path)
suv_data = pandas.read_csv('SUV Sales Analysis.csv')
#print(suv_data.info())
print(suv_data.head())

#logistic regression
x = suv_data.iloc[:, [2,3]].values  #independent variable
y = suv_data.iloc[:, 4].values  #dependent values

#divide the dataset into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#Scaling the dataset
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Logistic regression
classifier = LogisticRegression(random_state=0)
print(classifier.fit(x_train, y_train))

#Prediction
y_predict = classifier.predict(x_test)
print(accuracy_score(y_test, y_predict))

