import pandas
import matplotlib.pyplot as plt
import numpy
import os

os.chdir('/Users/harshshetye/Desktop/Machine Learning/Datasets')
data = pandas.read_csv('sample2.csv')
print(data.columns)
x=data['A'].values.tolist()
y=data['B'].values.tolist()
a=data['B'].values.tolist()
print(y)
#y = [i for i in y if i < 7]
y[:] = [i if i < 7 else 6 for i in y]
print(y)
plt.plot(x,a, label='Original Dataset')
plt.plot(x,y, label='Modified Dataset')
plt.xlabel('Time (unit)')
plt.ylabel('Power Consumed')
plt.legend()
plt.show()
