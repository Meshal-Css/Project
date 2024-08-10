#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


# Import dataset
df = pd.read_csv('Live.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df.drop(['Column1','Column2','Column3','Column4'],axis = 1,inplace = True)


# In[11]:


df.head()


# In[12]:


df.duplicated().sum()


# In[13]:


df.describe()


# In[16]:


df.corr


# In[17]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True)


# In[18]:


df.columns


# In[19]:


df.drop(['status_id'],axis = 1 , inplace = True)


# In[20]:


df.columns


# In[23]:


df['status_type'].unique()


# In[24]:


sns.countplot(x='status_type', hue='num_angrys', data=df)


# In[25]:


sns.countplot(x = 'status_type',  data = df)


# In[26]:


len(df['status_type'].unique())


# In[27]:


df['status_published'].unique()


# In[28]:


len(df['status_published'].unique())


# In[29]:


df['num_reactions'].unique()


# In[30]:


len(df['num_reactions'].unique())


# In[31]:


df.drop(['status_published'],axis = 1 , inplace = True)


# In[32]:


df


# In[33]:


X = df
y = df['status_type']


# In[34]:


# convert
from sklearn.preprocessing import LabelEncoder


# In[35]:


lm = LabelEncoder()


# In[36]:


X['status_type'] = lm.fit_transform(X['status_type']) # طبق نفسه على ال y


# In[37]:


X.head()


# In[38]:


y


# In[39]:


y = lm.transform(y)


# In[40]:


y


# In[41]:


# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
Mx = MinMaxScaler()


# In[44]:


X = Mx.fit_transform(X)


# In[50]:


X


# In[52]:


# kmeans
from sklearn.cluster import KMeans


# In[53]:


kmeans = KMeans(n_clusters=2, random_state=0)


# In[54]:


kmeans.fit(X)


# In[55]:


kmeans.cluster_centers_


# In[56]:


kmeans.inertia_


# In[57]:


# 14. Check quality of weak classification by the model 

labels = kmeans.labels_


# In[58]:


correct_labels = sum(y == labels)


# In[59]:


print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))


# In[60]:


print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[61]:


from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)


# In[62]:


plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[65]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3,random_state=0) # غير الرقم واحسب n_clusters

kmeans.fit(X)

labels = kmeans.labels_


# In[66]:


correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[ ]:




