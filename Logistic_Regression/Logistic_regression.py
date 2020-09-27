#We will determine the factors for survival in titanic by logistic regression
#Titanic data analysis
import numpy
import pandas
import seaborn as sns
import os
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

pandas.set_option('display.max_columns', 30)
pandas.set_option('display.max_rows', 160)
os.chdir('/Users/harshshetye/Desktop/Machine_Learning/Datasets')
titanic_data = pandas.read_csv('titanic2.csv')

#Count plot
#sns.countplot(x = 'Survived', data = titanic_data)

#Count plot with hue
#sns.countplot(x='Fare', hue='Sex', data = titanic_data)
#sns.countplot(x = 'Survived', hue='Pclass', data = titanic_data)

#titanic_data[''].plot.hist()
#plt.show()

#Data Wrangling To check and clean the Nan values
#print(titanic_data.isnull())
#print(titanic_data.isnull().sum())

'''sns.heatmap(titanic_data.isnull())
plt.show()'''
titanic_data.dropna(inplace=True)

#TO GE THE DUMMY VALUES IN PLACE OF STRINGS
sex = pandas.get_dummies(titanic_data['Sex'], drop_first=True) #without drop_first we will get two columns one for male and other for female
#print(sex.head(5))

embark = pandas.get_dummies(titanic_data['Embarked'], drop_first=True)
#print(embark)

Pcl = pandas.get_dummies(titanic_data['Pclass'], drop_first=True)
#print(Pcl)

#NOW TO CONCAT THESE DUMMY DATA WITH THE DATASET
titanic_data = pandas.concat([titanic_data, sex, embark, Pcl], axis=1)


titanic_data.drop(['Sex', 'Embarked', 'PassengerId', 'Ticket', 'Name', 'Pclass', 'Cabin'], axis=1, inplace=True)

#--------------------------------------------------------------------
#TRAINING AND TESTING DATASET
#--------------------------------------------------------------------
x = titanic_data.drop('Survived', axis=1) #This is the independent variable and user everything except the survived column
y = titanic_data['Survived'] #It is to predict if the passenger survived or not, Depedent variable

#Splitting the dataset into testing and training dataset
#For this 'from sklearn.model_selection import train_test_split' is used
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
logmodel = LogisticRegression() #Instance of logistic regression
#fit the model with the training dataset
print(logmodel.fit(x_train, y_train))

#Making Prediction
prediction = logmodel.predict(x_test)
print(x_test.head())

print('classification report', classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))
print(accuracy_score(y_test, prediction))