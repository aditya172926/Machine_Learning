import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import time
import csv
'''
def animate(i):
    ax1.clear()
    ax1.plot(x,y)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

print(os.getcwd())
os.chdir('/Users/harshshetye/Desktop/Machine Learning/Datasets')
data = pandas.read_csv('sample2.csv')
x = data['A'].values
y = data['B'].values

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
'''
os.chdir('/Users/harshshetye/Desktop/Machine Learning/Datasets')
path = 'sample2.csv'
with open(path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row[0])



    
