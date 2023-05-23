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


hospital_appointment = pd.read_csv("C:/Users/VDT/OneDrive/Desktop/hospital_appointment.csv")


# In[3]:


hospital_appointment.head()


# In[4]:


hospital_appointment.tail(10)


# In[5]:


hospital_appointment.info()


# In[6]:


hospital_appointment.columns


# In[7]:


hospital_appointment.describe()


# In[8]:


hospital_appointment.shape


# In[9]:


hospital_appointment.rename(columns={'No-show':'No_show'},inplace=True)
hospital_appointment.rename(columns={'Hipertension':'Hypertension'}, inplace=True)
hospital_appointment.rename(columns={'Handcap':'Handicap'}, inplace=True)


# In[10]:


hospital_appointment.head()


# In[11]:


hospital_appointment.isnull().sum()


# In[12]:


hospital_appointment.nunique()


# In[13]:


hospital_appointment1= hospital_appointment[['Gender','Scholarship','Hypertension','Diabetes','Alcoholism','Handicap','SMS_received','No_show']]


# In[14]:


plt.figure(figsize=(10,5))
sns.countplot('Scholarship',data = hospital_appointment,palette='hls')
plt.xticks(rotation =60)
plt.show()


# In[ ]:


plt.figure(figsize =(10,5))
sns.countplot('Age',data=hospital_appointment,palette='hls')
plt.xticks(rotation=70)
plt.show()


# In[ ]:


sns.distplot(hospital_appointment['Diabetes'])


# In[ ]:


sns.jointplot(hospital_appointment['Diabetes'],hospital_appointment['Age'])


# In[ ]:


sns.jointplot(hospital_appointment['Diabetes'],hospital_appointment['Age'],kind ="hex")


# In[ ]:


sns.jointplot(hospital_appointment['Diabetes'],hospital_appointment['Age'],kind ="kde")


# In[ ]:


sns.pairplot(hospital_appointment[['Diabetes','Age','Alcoholism']])


# In[ ]:


sns.stripplot(hospital_appointment['Diabetes'],['Handicap'])

