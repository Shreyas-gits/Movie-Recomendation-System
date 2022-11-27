#!/usr/bin/env python
# coding: utf-8

# # Project Overview
# -----------------------------------------
# Objective of the project is to build an app with a simple UI. This app will allow the user to search for movies and recommendations.
# 
# Different methods for creating recommendations system: 
# 1) Collaborative Filtering.
# 2) Content-based Filtering.
# 3) Personalized Video Ranker.
# 4) Candidate Generation Network.
# 5) Knowledge-based Recommender systems.
# 
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Meta Data

# #### Ratings Data File Structure (ratings.csv)
# -----------------------------------------
# 
# All ratings are contained in the file `ratings.csv`. Each line of this file after the header row represents one rating of one movie by one user, and has the following format:
# 
#     userId,movieId,rating,timestamp
# 
# The lines within this file are ordered first by userId, then, within user, by movieId.
# 
# Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).
# 
# Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.
# 
# 
# #### Tags Data File Structure (tags.csv)
# -----------------------------------
# 
# All tags are contained in the file `tags.csv`. Each line of this file after the header row represents one tag applied to one movie by one user, and has the following format:
# 
#     userId,movieId,tag,timestamp
# 
# The lines within this file are ordered first by userId, then, within user, by movieId.
# 
# Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user.
# 
# Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.
# 
# 
# #### Movies Data File Structure (movies.csv)
# ---------------------------------------
# 
# Movie information is contained in the file `movies.csv`. Each line of this file after the header row represents one movie, and has the following format:
# 
#     movieId,title,genres
# 
# Movie titles are entered manually or imported from <https://www.themoviedb.org/>, and include the year of release in parentheses. Errors and inconsistencies may exist in these titles.
# 
# Genres are a pipe-separated list, and are selected from the following:
# 
# * Action
# * Adventure
# * Animation
# * Children's
# * Comedy
# * Crime
# * Documentary
# * Drama
# * Fantasy
# * Film-Noir
# * Horror
# * Musical
# * Mystery
# * Romance
# * Sci-Fi
# * Thriller
# * War
# * Western
# * (no genres listed)
# 
# 
# #### Links Data File Structure (links.csv)
# ---------------------------------------
# 
# Identifiers that can be used to link to other sources of movie data are contained in the file `links.csv`. Each line of this file after the header row represents one movie, and has the following format:
# 
#     movieId,imdbId,tmdbId
# 
# movieId is an identifier for movies used by <https://movielens.org>. E.g., the movie Toy Story has the link <https://movielens.org/movies/1>.
# 
# imdbId is an identifier for movies used by <http://www.imdb.com>. E.g., the movie Toy Story has the link <http://www.imdb.com/title/tt0114709/>.
# 
# tmdbId is an identifier for movies used by <https://www.themoviedb.org>. E.g., the movie Toy Story has the link <https://www.themoviedb.org/movie/862>.
# 
# Use of the resources listed above is subject to the terms of each provider.
# 
# ---------------------------------------

# ### Importing Data

# In[2]:


df_movies=pd.read_csv('movies.csv')
df_movies.head()


# In[3]:


df_links=pd.read_csv('links.csv')
df_links.head()


# In[4]:


df_ratings=pd.read_csv('ratings.csv')
df_ratings.head()


# In[5]:


df_tags=pd.read_csv('tags.csv')
df_tags.head()


# In[6]:


print("Shape of Dataframes: \n"+ " Rating DataFrame"+ str(df_ratings.shape)+"\n Movies DataFrame"+ str(df_movies.shape)+"\n Tags DataFrame"+str(df_tags.shape)+"\n Links DataFrame"+str(df_links.shape))


# -----------------------------------------
# ### Merging datasets
# -----------------------------------------

# In[7]:


df_rating_movies=pd.merge(df_movies,df_ratings,on='movieId')
df_rating_movies=df_rating_movies.drop('timestamp',axis=1)
df_rating_movies.head()


# In[8]:


df_tags_movies=pd.merge(df_movies,df_tags,on='movieId')
df_tags_movies=df_tags_movies.drop('timestamp',axis=1)
df_tags_movies.head()


# In[9]:


# print("Shape of Ratings and movies merged dataset "+str(df_rating_movies.shape))
# print("Shape of Tags and movies merged dataset "+str(df_tags_movies.shape))


# -----------------------------------------
# ## Data Visualization
# -----------------------------------------

# #### Average rating and Number of ratings for each movies
# -----------------------------------------

# In[10]:


ratings = pd.DataFrame(df_rating_movies.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df_rating_movies.groupby('title')['rating'].count())
ratings


# In[11]:


# print('Max number of rating for a movie: ',ratings['num of ratings'].max())
# print('Min number of rating for a movie: ',ratings['num of ratings'].min())


# In[12]:


# plt.figure(figsize=(10,4))
# ratings['num of ratings'].hist(bins=50)
# plt.title('Distribution of num of ratings')
# plt.show()


# In[13]:


# print('Max average rating for a movies: ',ratings['rating'].max())
# print('Min average rating for a movie: ',ratings['rating'].min())


# In[14]:


# plt.figure(figsize=(10,4))
# ratings['rating'].hist(bins=11)
# plt.title('Distribution of average ratings')
# plt.show()


# In[15]:


# plt.figure(figsize=(10,4))
# sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
# plt.show()


# In[16]:


# plt.figure(figsize=(20,7))
# generlist = df_rating_movies['genres'].apply(lambda generlist_movie : str(generlist_movie).split("|"))
# geners_count = {}

# for generlist_movie in generlist:
#     for i in generlist_movie:
#         if(geners_count.get(i, False)):
#             geners_count[i]=geners_count[i]+1
#         else:
#             geners_count[i] = 1       
# geners_count.pop("(no genres listed)")
# plt.bar(geners_count.keys(),geners_count.values(),color='y')
# plt.show()


# In[17]:


# plt.figure(figsize=(10,4))
# sns.distplot(df_rating_movies["rating"])
# plt.title('Density plot of rating')
# plt.show()


# In[18]:


# ratings_grouped_by_users = df_rating_movies.groupby('userId').agg([np.size, np.mean])
# ratings_grouped_by_users = ratings_grouped_by_users.drop('movieId', axis = 1)
# ratings_grouped_by_users['rating']['size'].sort_values(ascending=False).head(10).plot.bar(figsize = (10,5))
# plt.title('Users who gave most number of ratings')
# plt.show()


# In[19]:


# plt.figure(figsize=(20,7))
# ratings_grouped_by_movies = df_rating_movies.groupby('title').agg([np.mean, np.size])
# ratings_grouped_by_movies.shape
# ratings_grouped_by_movies['rating']['size'].sort_values(ascending=False).head(10).plot.bar( figsize = (10,5))
# plt.title('Movies with most number of ratings')
# plt.show()


# -----------------------------------------
# ## Building movie based recomendation system.
# -----------------------------------------

# In[20]:


from scipy.sparse import csr_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity


# In[21]:


def encode(series, encoder):
    return encoder.fit_transform(series.values.reshape((-1, 1))).astype(int).reshape(-1)

user_encoder, movie_encoder = OrdinalEncoder(), OrdinalEncoder()
df_rating_movies['user_id_encoding'] = encode(df_rating_movies['userId'], user_encoder)
df_rating_movies['movie_id_encoding'] = encode(df_rating_movies['movieId'], movie_encoder)

matrix = csr_matrix((df_rating_movies['rating'], (df_rating_movies['user_id_encoding'], df_rating_movies['movie_id_encoding'])))


# In[22]:


matrix.shape


# In[23]:


df_rating_movies.head()


# In[24]:


df_matrix = pd.DataFrame(matrix.toarray())


# #### Normalizing the matrix <br>
# Rows represent Users <br>
# Columns represent Movies

# In[25]:


df_matrix = df_matrix.sub(df_matrix.sum(axis=1)/df_matrix.shape[1],axis=0)


# In[26]:


# df_matrix


# In[27]:


cosine_matrix = cosine_similarity(df_matrix.T)


# In[28]:


# pd.DataFrame(cosine_matrix)


# In[29]:


title_list = df_rating_movies.groupby('title')['movieId'].agg('mean')


# In[30]:


offline_results = {
    movie_id: np.argsort(similarities)[::-1]
    for movie_id, similarities in enumerate(cosine_matrix)
}
class recc:
    def get_recommendations(self,movie_title, top_n):
        movie_id = title_list[movie_title]
        movie_csr_id = movie_encoder.transform([[movie_id]])[0, 0].astype(int)
        rankings = offline_results[movie_csr_id][:top_n]
        ranked_indices = movie_encoder.inverse_transform(rankings.reshape((-1, 1))).reshape(-1)
        temp_df2 = df_movies.set_index('movieId').loc[ranked_indices]
        return list(temp_df2['title'])


# In[31]:


a = recc()
# a.get_recommendations('Toy Story (1995)',10)


# In[32]:


import pickle
pickle_out = open('recc.pkl', 'wb')
pickle.dump(a, pickle_out)
pickle_out.close()


# -----------------------------------------
# ## Building movie User recomendation system.
# -----------------------------------------

# In[33]:


cosine_matrix2 = cosine_similarity(df_matrix)


# In[34]:


# pd.DataFrame(cosine_matrix2)


# In[35]:


offline_results = {
    user_id: np.argsort(similarities)[::-1]
    for user_id, similarities in enumerate(cosine_matrix2)
}
class recc2:
    def get_user_recommendations(self,user_id, top_n):
        user_id = int(user_id)
        rankings = offline_results[user_id][1:top_n]
        ranked_indices = user_encoder.inverse_transform(rankings.reshape((-1, 1))).reshape(-1)
        temp_df = df_rating_movies.set_index('userId').loc[ranked_indices].sort_values('rating', ascending=False).iloc[:10,:].drop(['movieId','genres','rating','user_id_encoding','movie_id_encoding'], axis=1)
        return list(temp_df['title'])


# In[38]:


b = recc2()
# b.get_user_recommendations(15,2)


# In[37]:


pickle_out2 = open('recc2.pkl', 'wb')
pickle.dump(b, pickle_out2)
pickle_out2.close()

