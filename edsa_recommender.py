"""
    Streamlit webserver-based Recommender Engine.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.
    NB: !! Do not remove/modify the code delimited by dashes !!
    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------
    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.
	For further help with the Streamlit framework, see:
	https://docs.streamlit.io/en/latest/
"""

# Script dependencies
import pandas as pd
import numpy as np
import scipy as sp
import pickle
import copy
from surprise import Reader, Dataset, SVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Streamlit dependencies
import streamlit as st
import streamlit.components.v1 as components

# Data handling dependencies
import pandas as pd
import numpy as np
import codecs

# Custom Libraries
from utils.data_loader import (load_movie_titles)
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model


# Data Loading
movies = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)
title_list = load_movie_titles('resources/data/movies.csv')


# Content Based Model
def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.
    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.
    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.
    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset


def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.
    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.
    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.
    """
    # Initializing the empty list of recommended movies
    data = data_preprocessing(2000) ## CHANGE SUBSET TO MATCH RANGE IN APP
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data['keyWords'])
    names = data.copy()
    names.set_index('movieId',inplace=True)
    indices = pd.Series(names['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    cosine_sim = pd.DataFrame(cosine_sim, index = data['movieId'].values.astype(int), columns = data['movieId'].values.astype(int))
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_1).append(score_series_2).append(score_series_3).sort_values(ascending = False)

    listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending = False)
    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
    return recommended_movies


# Collaborative Based Model
# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.
    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.
    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.
    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.
    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.
    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.
    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store


def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.
    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.
    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.
    """
    names = movies_df.copy()
    names.set_index('movieId',inplace=True)
    indices = pd.Series(names['title'])
    users_ids = pred_movies(movie_list)
    # Get movie IDs and ratings for top users
    df_init_users = ratings_df[ratings_df['userId']==users_ids[0]]
    for i in users_ids[1:]:
        df_init_users = df_init_users.append(ratings_df[ratings_df['userId']==i])
    # Include predictions for chosen movies
    for j in movie_list:
        a = pd.DataFrame(prediction_item(j))
        for i in set(df_init_users['userId']):
            mid = indices[indices == j].index[0]
            est = a['est'][a['uid']==i].values[0]
            df_init_users = df_init_users.append(pd.Series([int(i),int(mid),est], index=['userId','movieId','rating']), ignore_index=True)
    # Remove duplicate entries
    df_init_users.drop_duplicates(inplace=True)
    #Create pivot table
    util_matrix = df_init_users.pivot_table(index=['userId'], columns=['movieId'], values='rating')
    # Fill Nan values with 0's and save the utility matrix in scipy's sparse matrix format
    util_matrix.fillna(0, inplace=True)
    util_matrix_sparse = sp.sparse.csr_matrix(util_matrix.values)
    # Compute the similarity matrix using the cosine similarity metric
    user_similarity = cosine_similarity(util_matrix_sparse.T)
    # Save the matrix as a dataframe to allow for easier indexing
    user_sim_df = pd.DataFrame(user_similarity, index = util_matrix.columns, columns = util_matrix.columns)
    user_similarity = cosine_similarity(np.array(df_init_users), np.array(df_init_users))
    user_sim_df = pd.DataFrame(user_similarity, index = df_init_users['movieId'].values.astype(int), columns = df_init_users['movieId'].values.astype(int))
    # Remove duplicate rows from matrix
    user_sim_df = user_sim_df.loc[~user_sim_df.index.duplicated(keep='first')]
    # Transpose matrix
    user_sim_df = user_sim_df.T
    # Find IDs of chosen load_movie_titles
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = user_sim_df[idx_1]
    rank_2 = user_sim_df[idx_2]
    rank_3 = user_sim_df[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Appending the names of movies
    listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending = False)
    # Choose top 50
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    # Get titles of recommended movies
    recommended_movies = []
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies_df[movies_df['movieId']==i]['title']))
    # Return list of movies
    recommended_movies = [val for sublist in recommended_movies for val in sublist]
    return recommended_movies


# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
	page_options = ["Recommender System","Solution Overview","EDA", "About The Team"]
	
    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
	page_selection = st.sidebar.selectbox("Choose Option", page_options)
	if page_selection == "Recommender System":
		# Header contents
		st.write('# Movie Recommender Engine')
		st.write('### EXPLORE Data Science Academy Unsupervised Predict')
		st.image('resources/imgs/Image_header.png',use_column_width=True)
		# Recommender System algorithm selection
		sys = st.radio("Select an algorithm",
			       ('Content Based Filtering',
				'Collaborative Based Filtering'))

			# User-based preferences
		st.write('### Enter Your Three Favorite Movies')
		movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
		movie_2 = st.selectbox('Second Option',title_list[25055:25255])
		movie_3 = st.selectbox('Third Option',title_list[21100:21200])
		fav_movies = [movie_1,movie_2,movie_3]
	
        # Perform top-10 movie recommendation generation
	sys = st.sidebar.selectbox("Recommender: ", [
                                       "Content Based Filtering", "Collaborative Based Filtering"])
	if sys == 'Content Based Filtering':
		if st.button("Recommend"):
			try:
				with st.spinner('Crunching the numbers...'):
					top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
					st.title("We think you'll like:")
					for i,j in enumerate(top_recommendations):
						st.subheader(str(i+1)+'. '+j)
			except:
				st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")
				

	if sys == 'Collaborative Based Filtering':
		if st.button("Recommend"):
			try:
				with st.spinner('Crunching the numbers...'):
					top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
					st.title("We think you'll like:")
					for i,j in enumerate(top_recommendations):
						st.subheader(str(i+1)+'. '+j)
			except:
				st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")
				
				
    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
		if page_selection == "Solution Overview":
			st.title("Solution Overview")
			st.markdown("Around the world, movie industries have been blessed with creative geniuses in the form of directors,   \
            screenwriters, actors, sound designers and cinematographers. Together with the rise in popularity of portable    \
            devices, capable of hosting streaming services, movies have ensured that people can stay glued to their          \
            favourites whether in transit or in the corners of their homes. \
            \
            \
            However, the spread into a plethora of genres ranging from romance to comedy to science fiction to horror has     \
            created a new problem of information overload, where choice and decision-making for individuals has become        \
            quite challenging.\
            \
            \
            In today’s technology driven world, there have been several attempts to solving this problem using recommender \
            systems. These systems are basically a subclass of intelligent information filtering processes that provide    \
            suggestions for items that are most pertinent to a particular user.")
			st.subheader("Approach")
			st.markdown("In this project, the **Infinity AI** team identifies some insights into data that can be used for the \
			development of a few recommender systems. The team explores eight datasets of more than 48000 movies and over   \
			160000 users with up to 15 million of datapoints containing movie ratings, genres, keywords, and so on collected   \
			from Explore Ai Academy (EDSA) and the MovieLens datasets. Using these datasets, the team attempts to answer       \
			various questions about movies. Delivering an accurate and robust solution to this challenge has immense economic  \
			potential for industry clients, with users of the system being exposed to content they would like to view or       \
			purchase - generating revenue and platform affinity.")
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
	
		if page_selection == "About The Team":
			st.title("About The Team")

			st.markdown("Infinity AI is a leading data science firm and AI-powered solutions provider with its own R&D Center. \
			\
			\
			Infinity AI’s mission is to help clients improve competitiveness and get great results from deploying tools in their \
			businesses. The team strives to reach these goals by applying innovative and proprietary development technologies,   \
			providing exceptional services, and using excellent professional expertise.                                          \
			\
			\
			Infinity AI is reputed for helping businesses get valuable insights from data, support customer retention while      \
			minimizing churn, optimize earnings, automate repetitive tasks, enhance performance, add AI-driven features, and     \
			prevent cost overruns.")

			st.write("Roles and Responsibilities of the Team")
			st.subheader("Meet the Team")
			if st.button('Bongani'): # information is hidden if button is clicked
				st.markdown('Bongani Mavuso is a Infinity AI CEO')
			if st.button('Aniedi'): # information is hidden if button is clicked
				st.markdown('Aniedi Oboho-Etuk is the Infinity AI Developer')
			if st.button('Tshepiso'): # information is hidden if button is clicked
				st.markdown('Tshepiso Padi is a Infinity Project Manager')
			if st.button('Josiah'): # information is hidden if button is clicked
				st.markdown('Josiah Aramide is the Infinity AI Developer/Strategist')
			if st.button('Manoko'): # information is hidden if button is clicked
				st.markdown('Manoko Langa is the Infinity AI Communications Manager')
			if st.button('Justice'): # information is hidden if button is clicked
				st.markdown('Ndinnanyi Justice is the Infinity AI Sales Manager')
			if st.button('Nsika'): # information is hidden if button is clicked
				st.markdown('Nsika Masondo is the Infinity AI Quality Control Manager')
	# -------------------------------------------------------------------
	
	# ------------- SAFE FOR ALTERING/EXTENSION -------------------------
	
		if page_selection == "Solution Overview":
			st.title("Solution Overview")
		st.write("Describe your winning approach on this page")

		
	# You may want to add more sections here for aspects such as an EDA,
    	# or to provide your business pitch.


if __name__ == '__main__':
    main()
