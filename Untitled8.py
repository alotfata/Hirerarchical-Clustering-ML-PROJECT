#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv("/Users/aynaz/Desktop/Mall_Customers.csv")
#X = dataset.iloc[:, [3, 4]].values


# In[18]:


dataset


# In[3]:


x=dataset.iloc[:, [3,4]].values


# In[7]:


import scipy.cluster.hierarchy as sch


# In[11]:


dendrogram= sch.dendrogram(sch.linkage(x, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Eucleadian Disatance")
plt.show()


# In[13]:


from sklearn.cluster import AgglomerativeClustering


# In[16]:


hc=AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward", distance_threshold=None)


# In[20]:


y_hc=hc.fit_predict(x)


# In[21]:


plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




