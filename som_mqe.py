#!/usr/bin/env python
# coding: utf-8

# In[79]:


"""
som_mqe.ipynb
Created at 08/06/2019
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sompy.sompy import SOMFactory

# Change this project dir
projectDir = "~/Desktop/HW3"


# In[80]:


"""
Normalize the data points to unit vector
param points: An array of 2D points
"""
def normalization(points):
    pointsSquare = np.square(points)
    lengths = pointsSquare[:, 0] + pointsSquare[:, 1]
    lengths = np.square(lengths)
    
    points[:, 0] / lengths
    points[:, 1] / lengths
    
    return points


# In[81]:


# read the data
featurePath = os.path.expanduser(projectDir)

trainingDataframe = pd.read_csv(featurePath + "/data_feature.csv", header = None)
trainingDataframe.columns = ["20.8Hz", "84.6Hz", "label"]
trainingDataframe.drop(columns = "label", inplace = True)

# Show the data a bit
# dataframe


# In[82]:


# read testing data
testingDataframe = pd.read_csv(featurePath + "/data_testing.csv", header = None)
testingDataframe.columns = ["20.8Hz", "84.6Hz", "label"]
testingDataframe.drop(columns = "label", inplace = True)


# In[83]:


# We need only "Healthy" data for SOM-MQE
healthyDataframe = trainingDataframe[:20]
#healthyDataframe


# In[84]:


#healthyDataArray = healthyDataframe.values
# Normalization
healthyDataArray = normalization(healthyDataframe.values)


# In[85]:


# Train the data, normalization is implemented in the process of building SOM
mapSize = [5, 5]

sm = SOMFactory().build(healthyDataArray, mapSize, normalization = None, initialization = "random", component_names=["20.8Hz", "84.6Hz"])
sm.train(n_job=1, verbose = False, train_rough_len=2, train_finetune_len = 100)    # I left some of the codes as the example provided


# In[86]:


# plot the results, components map
from sompy.visualization.mapview import View2D

view2D = View2D(20, 20,"rand data",text_size=10)
view2D.show(sm, col_sz = 3, which_dim = "all", denormalize = False)


# In[87]:


# Hit maps
from sompy.visualization.bmuhits import BmuHitsView

vhts = BmuHitsView(8, 8, "Hits Map", text_size = 7)
vhts.show(sm, anotate = True, onlyzeros = False, labelsize = 12, cmap = "jet", logaritmic = False)


# In[88]:


trainingDataArray = trainingDataframe.values
testingDataArray = testingDataframe.values

trainingDataArray = normalization(trainingDataArray)
testingDataArray = normalization(testingDataArray)

#trainingDataArray


# In[89]:


"""
Calculates the distance between the input data and the nodess of the map
"""
def inputToNodeDistances(data):
    mapMatrixSquare = sm.codebook.matrix.copy()
    mapMatrixSquare -= data
    mapMatrixSquare = np.square(mapMatrixSquare)
    distances = mapMatrixSquare[:, 0] + mapMatrixSquare[:, 1]
    distances = np.sqrt(distances)
    print(distances)
    
    return distances


# In[90]:


trainingDistances = []
testingDistances = []

for trainData in trainingDataArray:
    trainingDistances.append(np.amin(inputToNodeDistances(trainData)))
    
for testData in testingDataArray:
    testingDistances.append(np.amin(inputToNodeDistances(testData)))

#trainingDistances


# In[139]:


# Plot 
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,9]
plt.rcParams["font.size"] = 22
plt.title("SOM MQE Value of Training Data")
plt.xlabel('Sample')
plt.ylabel('MQE Value')
plt.plot(trainingDistances)
plt.axhline(y = 0.003, color = 'orange', linestyle = ':')
plt.axhline(y = 0.025, color = 'red', linestyle = ':')
plt.show()


# In[141]:


plt.title("SOM MQE Value of Testing Data")
plt.xlabel('Sample')
plt.ylabel('MQE Value')
plt.plot(testingDistances)
plt.axhline(y = 0.003, color='orange', linestyle=':')
plt.axhline(y = 0.025, color='red', linestyle=':')
plt.show()


# In[ ]:




