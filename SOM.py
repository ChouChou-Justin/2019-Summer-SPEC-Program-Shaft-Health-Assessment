# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:58:44 2019

@author: Justin
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import apply_along_axis
from numpy import linalg
with open('C:\\Users\\Justin\\Desktop\\SPEC Homework3\\data_feature.csv', 'r') as f:
  reader = csv.reader(f)
  Dataset = list(reader)
feature1 = [float(Dataset[i][0]) for i in range(len(Dataset))]
feature2 = [float(Dataset[i][1]) for i in range(len(Dataset))]
label = [int(Dataset[i][2]) for i in range(len(Dataset))]
Alldata = []
for i in range(len(feature1)):
    Alldata.append([feature1[i],feature2[i]])
Alldata = np.array(Alldata)
data0 = [Alldata[i] for i in range(20)]
label0 = [label[i] for i in range(20)]
data1 = [Alldata[i] for i in range(20,40)]
label1 = [label[i] for i in range(20,40)]
data2 = [Alldata[i] for i in range(40,60)]
label2 = [label[i] for i in range(40,60)]
#------------------------------------------------------------------------------
from minisom import MiniSom
#som0
som0 = MiniSom(5, 5, 2, sigma=1, learning_rate=0.8)
som0.random_weights_init(data0)
starting_weights = som0.get_weights().copy()
som0.train_random(data0, 1000) # training with 100 iterations
plt.figure(figsize=(6, 6))
wmap = {}
im = 0
for x, t in zip(data0, label0):  # scatterplot
    w = som0.winner(x)
    wmap[w] = im
    plt. text(w[0]+.5,  w[1]+.5,  str(t),
              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
    im = im + 1
plt.axis([0, som0.get_weights().shape[0], 0,  som0.get_weights().shape[1]])
plt.show()
weight0 = som0._weights
end_weight=som0.get_weights().copy()
#som1
som1 = MiniSom(5, 5, 2, sigma=1, learning_rate=0.8)
som1.random_weights_init(data1)
som1.train_random(data1,1000) # training with 100 iterations
plt.figure(figsize=(6, 6))
wmap = {}
im = 0
for x, t in zip(data1, label1):  # scatterplot
    w = som1.winner(x)
    wmap[w] = im
    plt. text(w[0]+.5,  w[1]+.5,  str(t),
              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
    im = im + 1
plt.axis([0, som1.get_weights().shape[0], 0,  som1.get_weights().shape[1]])
plt.show()
weight1 = som1._weights
#som2
som2 = MiniSom(5,5,2,sigma=1, learning_rate=0.8)
som2.random_weights_init(data2)
som2.train_random(data2, 1000) # training with 100 iterations
plt.figure(figsize=(6, 6))
wmap = {}
im = 0
for x, t in zip(data2, label2):  # scatterplot
    w = som2.winner(x)
    wmap[w] = im
    plt. text(w[0]+.5,  w[1]+.5,  str(t),
              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
    im = im + 1
plt.axis([0, som2.get_weights().shape[0], 0,  som2.get_weights().shape[1]])
plt.show()
weight2 = som2._weights
weightlist = []
weightlist.append(weight0)
weightlist.append(weight1)
weightlist.append(weight2)
#print(weightlist)
#print(weightlist[1])
#print(weightlist[1][0])
#print(weightlist[1][0][0])
#------------------------------------------------------------------------------
with open('C:\\Users\\Justin\\Desktop\\SPEC Homework3\\data_testing.csv', 'r') as f:
  reader = csv.reader(f)
  Dataset2 = list(reader)
x1 = [float(Dataset2[i][0]) for i in range(len(Dataset2))]
x2 = [float(Dataset2[i][1]) for i in range(len(Dataset2))]
answer = [int(Dataset2[i][2]) for i in range(len(Dataset2))]
testdata = []
for i in range(len(x1)):
    testdata.append([x1[i], x2[i]])
testdata = np.array(testdata)

def getDistance(coordinate1, coordinate2):
    distance = 0
    distance = ((coordinate1[1] - coordinate2[1])**2 + (coordinate1[0] - coordinate2[0])**2)**0.5
    return distance
distancelist = []
result = []
for i in range(30):
    for j in range(3):          #0, 1, 2
        for k in range(5):      #0, 1, 2, 3, 4
            for l in range(5):  #0, 1, 2, 3, 4
                distance = getDistance(testdata[i], weightlist[j][k][l])
                distancelist.append(distance)
    mini = min(distancelist)
    for x in range(len(distancelist)):
        if distancelist[x] == mini:
            if (0 <= x) and (x <= 24):
                result.append(0)
            elif (25 <= x) and (x <= 49):
                result.append(1)
            elif (50 <= x) and (x <= 74):
                result.append(2)
    distancelist = []           
print('Result of SOM: ', result)
print('Answer to SOM: ', answer)