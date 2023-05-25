#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv("German_data.csv")


# In[3]:


df


# In[4]:


df.head(15)


# In[5]:


df.tail(15)


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


df.describe(include="all")


# In[9]:


df.info()


# In[8]:


df=pd.read_csv("German_data.csv")


# In[9]:


df.head(5)


# In[11]:


df.shape


# In[13]:


df["CreditRisk"].value_counts()


# In[14]:


df.groupby("CreditRisk").mean()


# In[16]:


X=df.drop(columns='CreditRisk',axis=1)
Y=df['CreditRisk']


# In[17]:


print(X)


# In[18]:


print(Y)


# In[35]:





# In[36]:





# In[ ]:





# In[ ]:




