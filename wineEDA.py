#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import plotly.express as px


# Problem Statement: Which wine attributes are related to customers rating wines as good vs. bad?

# In[2]:


import os
os.path.isfile('/Users/AnnaD/Desktop/wine')
df_winequalityred = pd.read_csv('/Users/AnnaD/Desktop/Brainstation/wine/winequality-red.csv', delimiter=';')
df_winequalitywhite = pd.read_csv('/Users/AnnaD/Desktop/Brainstation/wine/winequality-white.csv', delimiter=';')


# In[3]:


df_winequalityred.head()


# In[4]:


df_winequalityred = df_winequalityred.add_prefix('red_')
df_winequalityred


# In[5]:


fig = px.histogram(df_winequalityred,x='red_quality')

fig.show()


# In[6]:


corr_red = df_winequalityred.corr()['red_quality'].drop('red_quality')
print(corr_red)


# In[7]:


sns.heatmap(df_winequalityred.corr())
plt.show()


# In[8]:


df_winequalityred.hist(column=['red_quality'])


# In[9]:


df_winequalityred['red_quality'] = df_winequalityred['red_quality'].apply(lambda x: 1 if x >=6  else 0)


# In[10]:


df_winequalityred.hist(column=['red_quality'])


# In[11]:


df_winequalityred = df_winequalityred.drop_duplicates()


# In[12]:


df_winequalityred.isna().sum()


# In[13]:


df_winequalityred.duplicated().sum()


# In[ ]:





# In[14]:


redcorr = df_winequalityred.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 8))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(redcorr, cmap='viridis_r', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(redcorr.columns)), redcorr.columns);
#Apply yticks
plt.yticks(range(len(redcorr.columns)), redcorr.columns)
#show plot
plt.show()


# In[ ]:





# In[15]:


from sklearn.preprocessing import LabelEncoder
label_qualityred = LabelEncoder()


# In[16]:



#df_winequalityred['red_quality'] = label_qualityred.fit_transform(df_winequalityred['red_quality'])


# In[17]:


df_winequalityred['red_quality'].value_counts()


# In[18]:


#label_qualitywhite = LabelEncoder()
#df_winequalitywhite['white_quality'] = label_qualitywhite.fit_transform(df_winequalitywhite['white_quality'])


# In[19]:


Xr = df_winequalityred.drop('red_quality', axis = 1)

yr = df_winequalityred['red_quality']


# In[20]:


from sklearn.preprocessing import StandardScaler
Xr_features = Xr
#Xr = StandardScaler().fit_transform(Xr)


# In[21]:


Xr_train, Xr_test, yr_train, yr_test = train_test_split (Xr, yr, test_size = 0.2, random_state = 42)


# In[22]:


#decision tree

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
modelr = DecisionTreeClassifier(random_state=1)
modelr.fit(Xr_train, yr_train)
yr_pred1 = modelr.predict(Xr_test)
print(classification_report(yr_test, yr_pred1))


# In[23]:


#Random forest
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(Xr_train, yr_train)
yr_pred2 = model2.predict(Xr_test)
print(classification_report(yr_test, yr_pred2))


# In[24]:


#random forest results

feat_importances = pd.Series(model2.feature_importances_, index=Xr_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))


# In[25]:


feat_importances


# In[26]:


feat_importances = pd.Series(modelr.feature_importances_, index=Xr_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))


# In[ ]:
