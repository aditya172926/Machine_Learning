import pandas
import numpy
import matplotlib.pyplot as plt

pandas.set_option('display.max_columns', 4)
data = pandas.read_csv('Bicycle.csv')
print(data.head())