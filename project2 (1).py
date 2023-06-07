#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


book = pd.read_csv('Books.csv',encoding='Latin1')
book.head(2)


# In[3]:


Rating = pd.read_csv('Ratings.csv',encoding='Latin1')
Rating.head(3)


# In[4]:


User = pd.read_csv('Users.csv',encoding='Latin1')
User.head(3)


# In[5]:


print( "Shape of Books",book.shape)
print( "Shape of Users",User.shape)
print( "Shape of Ratings",Rating.shape)


# In[6]:


book.info()


# In[7]:


book.isnull().sum()


# In[8]:


plt.figure(figsize=(20,10))
plt.title("Heatmap for null values")
sns.heatmap(book.isnull(), cbar=True,cmap="YlGnBu",vmax= book.shape[0]) 


# In[9]:


book = book.dropna()


# In[10]:


book.shape


# In[11]:


book[book.duplicated()]


# In[12]:


Rating.shape


# In[13]:


Rating.info()


# In[14]:


Rating.isnull().sum()


# In[15]:


plt.figure(figsize=(20,10))
plt.title("Heatmap for null values")
sns.heatmap(Rating.isnull(), cbar=True,cmap="YlGnBu",vmin= 0,vmax= Rating.shape[0])


# In[16]:


Rating[Rating.duplicated()]


# In[17]:


User.shape


# In[18]:


User.info()


# In[19]:


User.isnull().sum()


# In[20]:


plt.figure(figsize=(20,10))
plt.title("Heatmap for null values")
sns.heatmap(User.isnull(), cbar=True,cmap="YlGnBu")


# In[21]:


User[User.duplicated()]


# In[22]:


outer_join = pd.merge(Rating, book, on ='ISBN',how= 'outer',indicator=True)
outer_join


# In[23]:


outer_join._merge.value_counts()


# In[24]:


Nan = outer_join[outer_join._merge.isin(['left_only','right_only'])]
Nan


# In[25]:


plt.figure(figsize=(20,10))
plt.title("Heatmap for null values")
sns.heatmap(Nan.isnull(), cbar=True,cmap="YlGnBu")


# In[26]:


Nan.isnull().sum()


# In[27]:


df = {'Data':["Inner Join Dataset","Outer Join Datset ","Left Only","Right Only" ,"Left  + Right","Both (Intersect)"],
      'Total Entries':["1031129","1150989","118651","1209","119860","1031129",]}
Table = pd.DataFrame(df)
Table.set_index('Data',inplace=True)
Table


# In[28]:


inner_join = pd.merge(Rating, book, on ='ISBN',how= 'inner')
inner_join


# In[29]:


inner_join1 = pd.merge(inner_join, User, on ='User-ID')
inner_join1


# In[30]:


inner_join1.info()


# # EDA

# In[31]:


round(inner_join1.describe())


# In[32]:


inner_join1.isnull().sum()


# ## Replacing coloumn(Age) nulls with mean

# In[33]:


mean_value=inner_join1['Age'].mean()
inner_join1['Age'].fillna(value = mean_value, inplace=True)
inner_join1.isnull().sum()


# In[34]:


inner_join1[inner_join1.duplicated()]


# In[35]:


inner_join1['Year-Of-Publication'].unique()


# In[36]:


(inner_join1.loc[inner_join1['Year-Of-Publication']== '0'])


# In[37]:


inner_join1 = inner_join1.loc[(inner_join1['Year-Of-Publication'] != '0')]


# In[38]:


plt.figure(figsize=(20,10))
plt.title("Heatmap for null values in merged dataset",fontsize = 15)
sns.heatmap(inner_join1.isnull(), cbar=True,cmap="YlGnBu",vmin= 0 , vmax = inner_join1.shape[0])


# In[39]:


plt.figure(figsize=(10,100))
ax = sns.barplot(x= inner_join1['Book-Title'].value_counts().iloc[0:200],y =inner_join1['Book-Title'].value_counts().iloc[0:200].index
           ,palette = "Greens_d")
ax.bar_label(ax.containers[0],fontsize = 15)

plt.title("Top Books In Dataset"
          ,color = "darkGreen",fontsize = 20)
plt.show()


# In[40]:


inner_join1_null = inner_join1[inner_join1['Book-Rating'] != 0]
inner_join1_null.head()


# In[41]:


inner_join1_null.shape


# In[42]:


plt.show()
plt.figure(figsize=(20,10))
plt.title("Book Ratings Without Zero Values"
          ,fontsize = 20)
sns.countplot(x= inner_join1_null['Book-Rating'])
plt.show()


# In[43]:


round(inner_join1.describe())


# In[44]:


location_null = inner_join1[inner_join1['Location'] != 'n/a']
location_null


# In[45]:


plt.figure(figsize=(20,50))
ax = sns.barplot(x= location_null['Location'].value_counts().iloc[0:50],y =location_null['Location'].value_counts().iloc[0:50].index
           ,palette = "RdYlGn")
ax.bar_label(ax.containers[0],fontsize = 10)
plt.title("Top 50 Locations",fontsize = 20)
plt.show()


# In[46]:


num_rating= inner_join1_null.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating


# In[47]:


avg_rating = inner_join1_null.groupby('Book-Title').mean()['Book-Rating'].round(1).reset_index()
avg_rating.rename(columns={'Book-Rating':'avg_rating'},inplace=True)
avg_rating


# In[48]:


final_rating = num_rating.merge(avg_rating,on='Book-Title')
final_rating 


# In[49]:


final_rating1 = final_rating.merge(inner_join1_null,on='Book-Title')
final_rating1


# In[50]:


Final_rating = final_rating1[final_rating1['num_ratings']>=50].sort_values('num_ratings',ascending=False)
Final_rating 


# In[51]:


popular_book_50 = Final_rating.drop_duplicates('Book-Title').head(50)
popular_book_50 


# ## Pivot table

# In[52]:


book_pivot = Final_rating.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
book_pivot 


# In[53]:


book_pivot.fillna(0,inplace = True)
book_pivot 


# In[54]:


from sklearn.metrics.pairwise import cosine_similarity


# In[55]:


similarity_scores = cosine_similarity(book_pivot)


# In[56]:


similarity_scores.shape


# In[57]:


def recommend(book_name):
    
    index = np.where(book_pivot.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = Final_rating[Final_rating['Book-Title'] == book_pivot.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data


# In[58]:


recommend('The Virgin Blue')


# In[59]:


recommend('Peace Like a River')


# In[60]:


import pickle
pickle.dump(popular_book_50,open('popular_book_50.pkl','wb'))
pickle.dump(book_pivot,open('book_pivot.pkl','wb'))
pickle.dump(Final_rating,open('final_rating.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))

