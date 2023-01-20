#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings('ignore')


# In[19]:


df = pd.read_csv('Salary.csv')
df.head(10)


# In[20]:


df.shape


# In[21]:


df.describe()


# In[22]:


df.isnull().sum()


# In[23]:


df.duplicated().sum()


# In[24]:


plt.figure(figsize=(10,3))
plt.title("Boxplot of salary Variable.")
sns.boxplot(data=df['Salary'],color='#8B1A1A',orient='horizontal');


# ## Model Building

# In[26]:


model=IsolationForest(n_estimators=20, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(df[['Salary']])


# In[30]:


df['scores']=model.decision_function(df[['Salary']])
df['anomaly']=model.predict(df[['Salary']])
df


# In[29]:


anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)


# There are 4 outliers in our dataset, which can be removed in subsequent stages of EDA for better accuracy and prediction.
