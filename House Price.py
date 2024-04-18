#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[59]:


dataset=pd.read_csv("E:\DS project\Project House Price\Houseprice.csv")
dataset


# In[60]:


dataset.isnull().sum()


# In[61]:


data=dataset.dropna()
data


# In[62]:


data.isnull().sum()


# In[63]:


X=data[['INDUS','AGE','RAD']]
Y=data[['MEDV']]


# In[64]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[65]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)


# In[66]:


Prediction= lin_reg.predict([[2.14,65,1]])
Prediction


# In[67]:


Y_pred = lin_reg.predict(X_test)


# In[68]:


mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)


# In[ ]:




