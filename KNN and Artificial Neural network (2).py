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


titanic_data.head()


# In[4]:


#Extracting Independent and dependent Variable  
x= titanic_data.iloc[:, [2,3]].values  
y= titanic_data.iloc[:, 4].values  


# In[5]:


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  


# In[6]:


#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  
# Make predictions on the test set
#y_pred = classifier.predict(x_test)

# Print the predicted labels
#print(y_pred)


# In[7]:


#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train) 
# Make predictions on the test set
y_pred = classifier.predict(x_test)

# Print the predicted labels
print(y_pred)


# In[8]:


#Predicting the test set result  
y_pred= classifier.predict(x_test) 
# Make predictions on the test set
y_pred = classifier.predict(x_test)

# Print the predicted labels
print(y_pred)


# In[9]:


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  
# Make predictions on the test set
y_pred = classifier.predict(x_test)

# Print the predicted labels
print(y_pred)


# In[10]:


# Visualizing the training set result
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('K-NN Algorithm (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[11]:


# Visualizing the training set result
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('K-NN Algorithm (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[12]:


# starting from this file its for Artificial neural network.
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score


# In[13]:


x_train, x_test, y_train, y_test= train_test_split(titanic_data.drop('Age', axis=1), titanic_data['Age'],test_size=0.2)

#Initialize the ANN model
model=Sequential()


# In[14]:


#Add the input layer and first hidden layer
model.add(Dense(units=32, activation='relu',
                input_dim=x_train.shape[1]))


# In[15]:


#Add a second layer hidden layer
model.add(Dense(units=32, ))


# In[16]:


#add output layer
model.add(Dense(units=1, activation='sigmoid'))


# In[17]:


#Compile the model
model.compile(loss='binary_crossentropy',
             optimizer='adam', metrics=['accuracy'])


# In[21]:


# Assuming x_train has shape (number_of_samples, 5)
x_train = np.random.rand(100, 5)
y_train = np.random.randint(2, size=(100, 1))

# Removing the extra column from the input data
x_train = x_train[:, :4]

# Creating the model
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)


# In[22]:


#Evaluate the model on the test data
loss, accuracy=model.evaluate(x_test, y_test)


# In[23]:


#print the accuracy score
print('Accuracy:', accuracy)

