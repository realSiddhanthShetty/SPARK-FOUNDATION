#!/usr/bin/env python
# coding: utf-8

# # GRIP DEC21  @ THE SPARK FOUNDATION ( COMPANY)
# 
# # DATA SCIENCE AND BUSINESS ANALYTICS
# 
# # AUTHOR : SIDDHANTH SHETTY
# 
# # TASK 1 : PREDICTION USING SUPERVISED MACHINE LEARNING
# 

# ## **Linear Regression with Python Scikit Learn**
#    In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# ## **Simple Linear Regression**
#   In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# **Description** : To predict the percentage of Student based on the Hours of Study

#  ## IMPORTING ALL REQUIRED THE LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## IMPORT DATASET

# In[2]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url) 

data.head(10) # Frist 10 rows


# ## DATA PROCESSING

# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.describe()


# # Checking for NULL Values

# In[6]:


data.isnull().sum()   


# ## DATA VISUALIZATION

# Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# # Plotting the distribution of scores

# In[7]:


data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Preparing the Data

# In[8]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# # Splitting the dataset into train and test sets
# 

# In[9]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# # Training the Algorithm using LINEAR REGRESSION

# In[10]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# # Plotting the regression line

# In[11]:


regressor.coef_


# In[12]:


regressor.intercept_


# In[13]:



line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line)
plt.show()


# # Making Predictions

# In[14]:


print(X_test)       # Testing data - In Hours
y_pred = regressor.predict(X_test)       # Predicting the scores


# # Comparing Actual output vs Predicted output

# In[15]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # Predicting the score of the student based on hours studied

# Here our aim is to predict the score of student if she/he studied for 9 hours

# In[16]:


hours=9
pred=regressor.predict([[hours]])
print("Number of hours = {}".format(hours))
print("Predicted score = {}".format(pred[0]))


# # Accuracy of the model

# In[17]:


np.round(regressor.score(X_test,y_test)*100,2)


# # Evaluating the model

# In[19]:


#evaluating using mean absolute error
from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 

#evaluating using R square error
print('r2 score:',metrics.r2_score(y_test, y_pred)) 


# In[ ]:




