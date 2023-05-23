#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


titanic_data=pd.read_csv("C:/Users/VDT/OneDrive/Desktop/titanic_data.csv")


# In[3]:


titanic_data.head(20)


# In[4]:


titanic_data.tail(10)


# In[5]:


titanic_data.describe()


# In[6]:


titanic_data.nunique()


# In[7]:


titanic_data.shape


# In[8]:


titanic_data.drop(['Passengerid', 'zero', 'zero.1', 'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7', 'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13', 'zero.14', 'zero.15', 'zero.16', 'Embarked', 'zero.17', 'zero.18'], axis='columns',inplace=True)


# In[9]:


titanic_data.head()


# In[10]:


titanic_data.rename(columns={'2urvived':'Survived'},inplace=True)


# In[11]:


titanic_data.head()


# In[12]:


titanic_data.drop(['Parch','sibsp'],axis='columns',inplace=True)# dropping two variables


# In[13]:


titanic_data.head(5)


# In[14]:


target=titanic_data.Survived
inputs=titanic_data.drop('Survived',axis='columns')


# In[1]:


target.head()


# In[16]:


inputs.dtypes


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(inputs, target, test_size=titanic_data('Survived'))


# In[18]:


sns.stripplot(['Age'],['Sex'])


# In[19]:


sns.pairplot(titanic_data[['Survived','Age','Fare']])


# In[20]:


sns.boxplot(titanic_data['Age'],titanic_data['Survived'],titanic_data['Pclass'])


# In[21]:


sns.pointplot(titanic_data['Age'],titanic_data['Survived'],hue=titanic_data['Pclass'])


# In[22]:


sns.lmplot(x="Survived",y="Pclass",data=titanic_data)


# In[23]:


sns.jointplot(titanic_data['Survived'],titanic_data['Age'],kind ="kde")


# In[2]:


import turtle
turtle.speed(0)
for i in range(600):
    turtle.forward(i*2)
    turtle.left(119)
turtle.done()

