#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn import datasets,linear_model,metrics
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Training Dataset

# In[2]:


House = pd.read_csv(r'E:\Projects\archive\kc_house_data.csv')
House


# # Description

# In[3]:


House.head()


# In[4]:


House.tail()


# In[5]:


House.info()


# In[6]:


House.shape


# In[7]:


House.describe()


# In[32]:


#check for any Missing Values
House.isnull().sum()


# In[8]:


House.isnull().sum()*100/House.shape[0]


# In[9]:


#Finding out which is a common house BedroomWise
House["bedrooms"].value_counts().plot(kind='bar')
plt.title('Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# In[10]:


#Visualizing the location of the houses based on latitude and longitude
plt.figure(figsize=(10,10))
sns.jointplot(x=House.lat.values, y = House.long.values, size = 18)
plt.xlabel('Latitude',fontsize = 18)
plt.ylabel('Longitude',fontsize = 18)
plt.show()
sns.despine


# ### looking at common factors which are affecting the price of the houses

# # Price vs Square feet 

# In[11]:


plt.scatter(House.price,House.sqft_living)
plt.title("Price x Square Ft.")


# # Price vs Longitude

# In[12]:


plt.scatter(House.price,House.long)
plt.title("Price x Location of Area")


# # Price vs Latitude

# In[13]:


plt.scatter(House.price,House.lat)
plt.title("Price x Location of Area")


# In[14]:


plt.scatter(House.bedrooms,House.price)
plt.title("Bedroom and price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine()


# ### Total sqft. including Basement vs Price

# In[16]:


plt.scatter((House['sqft_living']+House['sqft_basement']),House['price'])


# ### Total sqft. including Waterfront vs Price

# In[34]:


plt.scatter(House.waterfront,House.price)
plt.title("Waterfront vs Price")


# ### Total sqft. including Floors vs Price

# In[17]:


plt.scatter(House.floors,House.price)


# ### Total sqft. including Condition vs Price

# In[18]:


plt.scatter(House.condition,House.price)


# In[37]:


X1 = House.drop(['id', 'price'],axis=1)
X1.head()


# In[39]:


House.floors.value_counts().plot(kind='bar')


# In[40]:


plt.scatter(House.floors,House.price)


# In[41]:


plt.scatter(House.condition,House.price)


# In[42]:


plt.scatter(House.zipcode,House.price)
plt.title("pricey location by zipcode?")


# # Model Creation For House Prediction
# 

# ## Linear Regression

# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


reg = LinearRegression()


# In[85]:


labels = House['price']
conv_dates = [1 if values == 2023 else 0 for values in House.date]
House['date']= conv_dates


# # Splitting Dataset into Training and Testing

# In[86]:


from sklearn.model_selection import train_test_split


# In[87]:


x_train, x_test, y_train, y_test = train_test_split(train1,labels , test_size = 0.20, random_state = 30)


# In[88]:


reg.fit(x_train,y_train)


# In[102]:


#Accuracy using Linear Regression
reg.score(x_test,y_test)
from sklearn.metrics import r2_score
print("score : ",r2_score(y_test,y_pred))


# # Gradient Boost

# In[90]:



from sklearn import ensemble
Gb = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1)


# In[91]:


Gb.fit(x_train,y_train)


# In[104]:


#Accuracy Using Gradint Boost Regressor
Gb.score(x_test,y_test)


# In[93]:


y_pred = reg.predict(x_test)


# # Prediction of House Price

# In[96]:


from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(scale(X1))


# In[97]:


x=Gb.predict([x_test.iloc[-10]])
x


# In[99]:


y_test.iloc[-10]
y


# # House Prediction Success

# # Thank You

# In[ ]:




