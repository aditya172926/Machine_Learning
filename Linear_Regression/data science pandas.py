import numpy
import os
import pandas
import matplotlib.pyplot as plt #for visualization
import seaborn as sns #for visualization


pandas.set_option('display.max_columns', 45)
pandas.set_option('display.max_rows', 5000)
health_data = pandas.read_csv('data-society-health-status-indicators/data/demographics.csv')

health_copy = health_data.copy(deep=True)
#print(health_copy.columns)
#Creating frequency table
'''
frequency_table = pandas.crosstab(index = health_copy['chsi_state_name'], columns = 'count',
                                  dropna = True)'''

'''two_way_table = pandas.crosstab(index = health_copy[''],
                                columns = health_copy['population_size'],
                                dropna = True)'''



#Data Visualization
#removing the missing values
health_copy.dropna(axis=0, inplace=True) #axis=0 is rows and inplace=True is for to reflect the changes in the dataframe

#Creating a scatter plot
#Syntax plt.scatter(x axis variable, y axis variable, color)
'''plt.scatter(health_copy['population_size'], health_copy['asian'], c = 'red')
plt.title('Scatter plot for population size vs poverty')
plt.xlabel('Population size')
plt.ylabel('Population Density')
plt.show()'''

#Creating histogram: A graphical representation of the data using bars of dofferent heights
#Use histogram to represent frequency distribution of a data
'''plt.hist(health_copy['population_size'], color='green', edgecolor='red', bins=5)
plt.title('Population frequency')
plt.show()'''

#Bar plot: Presents a categorical data
'''count = [100, 1000, 2000]
population = ('population_density', 'min_population_density', 'max_population_density')
index = numpy.arange(len(population))

plt.bar(index, count, color=['red', 'blue', 'green'])
plt.title('Bar graph of population densities')
plt.show()'''

#Visualization using seaborn
sns.set(style="darkgrid")
sns.regplot(x=health_copy['population_density'], y=['poverty']) #regplot stands for regression plot





















