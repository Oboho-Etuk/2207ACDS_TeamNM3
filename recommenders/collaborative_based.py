"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
#ratings_df = ratings_df.merge(movies_df, on='movieId')

# Rapid cleaning and pruning of dataframes to use
movies_df.drop(['genres'], axis=1, inplace=True)
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/small_svd.pkl', 'rb'))

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
    #model.fit(a_train)

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False)) #removed zero
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
    #indices = pd.Series(movies_df['movieId'])

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

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
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
    # use movieId from list of movies
    # to return userIds from the result
    #indices = pd.Series(movies_df['title'])
    userIds = pred_movies(movie_list) # a list of userIds
    
    # get similar users from ratings dataframe
    df_init_users = ratings_df[ratings_df['userId']==userIds[0]]

    # add other users by merging dataframes - ratings and movies 
    df_init_users = pd.merge(df_init_users, movies_df, on="movieId", how="inner")
    def get_movieId(movie_list):
        movieId = []
        # loop the movie list
        for movie in movie_list:
            res = movieId.append(int(movies_df['movieId'][movies_df['title'] == movie]))
        return res
    
    # get new users with similar ratings
    new_users_Id = [450000, 450000, 450000]
    new_users_movieId = [get_movieId(movie_list)]
    new_users_title = [movie_list[0], movie_list[1], movie_list[2]]
    new_users_rating = [4.5, 5.0, 5.0]
    listK = ['userId', 'movieId', 'title', 'rating']
    listV = [new_users_Id, new_users_movieId, new_users_title, new_users_rating]
    # build dictionary of new users...new_users_dict = {}
    new_users_dict = dict(zip(listK, listV))
    
    # make dictionary into a dataframe and append to initial users with continuous index
    new_users_df = pd.DataFrame.from_dict(new_users_dict) 
    df_init_users = df_init_users.append(new_users_df, ignore_index=True) 

    # create a utility matrix of our pool of users
    util_matrix = df_init_users.pivot_table(index='userId', columns='title', values='rating')
    util_matrix.fillna(0, inplace=True)

    # develop cosine similarity for the utility matrix
    cosine_sim = cosine_similarity(util_matrix, util_matrix)
    
    # get the resulting similarity dataframe
    util_sim_df = pd.DataFrame(cosine_sim, index=util_matrix.columns, columns=util_matrix.columns )
    
    # get recommendations for N most similar movies
    def get_recs(movie_list, N):
        for movie in movie_list:
            sim_scores = list(enumerate(util_sim_df[movie]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            #sim_scores = sim_scores.sort_values(ascending=False)       #alternative if above fails
            sim_scores = sim_scores[1:N+1]
            top_movies = [sim_scores[title] for title in util_sim_df.index]
            return top_movies
    
    # store the movie titles recommended
    top_movies = []

    # calculate the cumulative score for each movie in the df
    for movie in get_recs(movie_list, N=10):
        if movie in movie_list:
            pass
        else:
            # append the movie title on the list
            top_movies.append(movie)
    # get the top n movies
    recommended_movies = top_movies[:top_n]
    # return the recommended movies
    return recommended_movies
