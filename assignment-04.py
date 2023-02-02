#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
data = pd.read_csv('data.csv')
data


# In[5]:


data.mean()


# In[6]:


data.median()


# In[7]:


data.mode()


# In[8]:


data.std()


# In[9]:


data.var()


# In[10]:


data.isnull().sum()


# In[11]:


data['Calories'] = data['Calories'].fillna(data['Calories'].mean())


# In[12]:


data.isnull().sum()


# In[13]:


result = data[['Duration','Maxpulse']].agg(['min','max','count','mean'])
result


# In[14]:


d1 = data[data['Calories'].between(500,1000)]
d1


# In[15]:


d2 = data[(data['Calories']>500) & (data['Pulse']<100)]
d2


# In[16]:


data_modified=data.drop('Maxpulse',axis=1)
data_modified


# In[17]:


data.drop('Maxpulse',axis=1)


# In[18]:


data["Calories"]=data["Calories"].astype(float).astype(int)
data


# In[19]:


plot = data.plot.scatter(x="Duration",y="Calories")


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
salary_data = pd.read_csv('Salary_Data.csv')
salary_data


# In[26]:


X = salary_data["YearsExperience"]
Y = salary_data["Salary"]
#X1 = [[i,x] for i, x in enumerate(X)]
#Y1 = [[i,y] for i, y in enumerate(Y)]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)


# In[27]:


regressor = LinearRegression() 
model = regressor.fit(X_train.values.reshape(-1, 1),Y_train.values.reshape(-1, 1))


# In[28]:


print(model.coef_)
print(model.intercept_)


# In[29]:


Y_predict = model.predict(X_test.values.reshape(-1,1))
plt.title("Salary/Years of XP")
plt.ylabel("Salary $")
plt.xlabel("Years")
plt.scatter(X_test,Y_test,color="blue",label="real data")
plt.plot(X_test,Y_predict,color="red",label="linear model")
plt.legend()
plt.show()


# In[30]:


mean_squared_error(Y_test, Y_predict)


# In[ ]:




