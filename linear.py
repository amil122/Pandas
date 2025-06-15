#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



# In[43]:


df=pd.read_csv('Salary_dataset.csv')


# In[44]:


df.drop('Unnamed: 0',axis=1)


# In[45]:


plt.scatter(df['YearsExperience'],df['Salary'])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')


# From above graph we can see that as years of experience increses salary also increses and whether to find it is positive or negative we use correlation

# In[46]:


df.corr()


# In[47]:


X=df[['YearsExperience']] ### independent feature always need to be in 2d or in data frame 


# In[48]:


X


# In[51]:


y=df['Salary']


# In[52]:


y


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[55]:


from sklearn.preprocessing import StandardScaler


# In[56]:


scaler=StandardScaler()


# In[57]:


X_train=scaler.fit_transform(X_train)
X_train


# In[58]:


X_test=scaler.transform(X_test)


# Applying simple linear regression

# In[59]:


from sklearn.linear_model import LinearRegression


# In[60]:


lr=LinearRegression()


# In[61]:


lr.fit(X_train,y_train)


# In[62]:


print('coefficient or slope:',lr.coef_)


# One unit movement in years of experience will cause 27151 values in salaray

# In[63]:


print("Intercept:",lr.intercept_)


# when x=0,at what value does it touches y-axis signifies the intercept

# Plot training data plot best fit line

# In[64]:


plt.scatter(X_train,y_train)
plt.plot(X_train,lr.predict(X_train))


# Prediction for test data
# 

# predicted salary output=Intercept+coef_(YearsExp)

# In[67]:


y_pred=lr.predict(X_test) 
y_pred


# In[68]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[69]:


mse=mean_squared_error(y_test,y_pred)
mse


# In[70]:


mae=mean_absolute_error(y_test,y_pred)
mae


# In[71]:


rmse=np.sqrt(mse)
rmse


# Calculating R2sqaure

# R2=1-SSR/SST
# R2=coefficient of determination,SSR=sum of squares of residuals,SST=total sum of sqaures

# In[72]:


from sklearn.metrics import r2_score


# In[41]:


score=r2_score(y_test,y_pred)
score


# Calculating Adjusted R2 score
# Adjusted R2=1-[(1-R2)*(n-1)(n-k-1)]
# R2:R2 of model,n:number of observations,k:number of predictor variables

# Multiple Linear Regression

# In[ ]:




