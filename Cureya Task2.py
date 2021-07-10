#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


df = pd.read_csv('data.csv')


# In[3]:


df.head()


# In[5]:


print("Total no. of rows in dataset = {}".format(df.shape[0]))


# In[6]:


print("Total no. of columns in dataset = {}".format(df.shape[1]))


# In[7]:


df.replace(',','',regex=True, inplace=True)


# # TRAINIG THE DATASET

# In[9]:


target_col = "Total"
X= df.loc[:,df.columns[3:9]]
y= df.loc[:, target_col]

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=42)


# # Linear Regression Using Statsmodels

# In[15]:


X_with_constant = sm.add_constant(X_train)


# In[16]:


model = sm.OLS(y_train.astype(float), X_with_constant.astype(float))
results = model.fit()
results.params


# In[17]:


print(results.summary())


# In[11]:


X_test = sm.add_constant(X_test)
y_pred = results.predict(X_test.astype(int))
y_test = np.array(y_test, dtype=int)
X_test = np.array(X_test, dtype=int)
y_pred = np.array(y_pred, dtype=int)
X_train = np.array(X_train, dtype=int)
residual = y_test - y_pred


# # Finding Variance

# In[12]:


vif = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
pd.DataFrame({'vif': vif[0:]}, index=X_train[2]).T
print(vif)
sns.displot(residual)


# In[13]:


fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)


# # Residual

# In[14]:


np.mean(residual)

fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(y_pred, residual)


print(residual)


# In[ ]:




