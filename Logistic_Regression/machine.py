#machine learning model
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

pandas.set_option('display.max_rows', 200)

path = 'iris (1).csv'
#names of the columns from the the datasets
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#define the dataset
dataset = pandas.read_csv(path, names=names)
#print(dataset)
'''frequency_table = pandas.crosstab(dataset['class'], columns='count')
print(frequency_table)'''
#print(dataset.describe())
#print(dataset.groupby('class').size())  #similar to the frequency table of the passed variable. In this case the 'class'

#plot
'''dataset.plot(kind='box', subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show()'''

#Histigram: This histogramis a bit different and has a different syntax than the plt.hist. It makes the histogram for the all the columns of the dataset where as the other syntax makes it only for the passed column index
'''dataset.hist()
plt.show()

#multivariate plot
#Again the syntax is a bit different from the other as this generates a whole matrix
scatter_matrix(dataset)
plt.show()'''
#------------------------------------------------------------------------------------------------------------
#Creating a model
#Create a validation dataset

'''array = dataset.values
x = array[:,0:4] #all the columns from 0 to 4
y = array[:, 4]
validation_size = 0.20
seed = 6
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)
#we split the dataset into 2 parts and use the 20% of it to train and then test the model
#the above code is used to split the dataset'''

#------------------------------------------------------------------------------------------------------------
#Making a test harness
'''scoring = 'accuracy'

#Spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDR', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)'''
#We get the accuracy of the different algorithms used here. Out of these we select the most accurate one
#----------------------------------------------------------------------------------------------
#stats and probability
