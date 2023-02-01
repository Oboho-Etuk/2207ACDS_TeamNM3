"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler # for standardization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, preprocessing)

# Importing data - movies, ratings and imdb
mov = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
imdb = pd.read_csv('resources/data/imdb_data.csv', sep = ',')


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
    # instantiate quick pre-processing: to merge datasets for more attributes
    mov['movieId'] = mov['movieId'].astype('int')
    imdb['movieId'] = imdb['movieId'].astype('int')
    # merge on movieId present in both mov and imdb datasets
    movies = mov.copy()
    ##movies = mov.merge(imdb, on='movieId')
    
    # clean data
    movies.dropna(inplace=True)

    # select columns from merged dataset - title_cast, director, plot_keywords and genres
    ##movies = movies[['title_cast', 'director', 'plot_keywords', 'genres']]
    ##elements = ['title_cast', 'director', 'plot_keywords', 'genres']
    ##for item in elements:
    ##    movies[item] = movies[item].str.lower().str.replace(' ', '').str.replace('|', ' ')
    
    # collect keywords to apply vectorizer
    ##movies['keyWords'] = movies['plot_keywords'] \
    ##                    + ' ' + movies['title_cast'] \
    ##                    + ' ' + movies['director']   \
    ##                    + ' ' + movies['genres']     \
    
    # Split genre data into individual words
    genres = movies['genres'].str.lower().str.replace(' ', '').str.replace('|', ' ')
    ##title_cast = movies['title_cast'].str.lower().str.replace(' ', '').str.replace('|', ' ')
    ##movies['keyWords'] = genres + ' ' + title_cast
    movies['keyWords'] = genres
    
    # obtain subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
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
    recommended_movies = []
    data = data_preprocessing(27000)
    ##data = data_preprocessing(12000)
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer(stop_words='english')
    count_matrix = count_vec.fit_transform(data['keyWords'])
    
    # Instantiate a scaler to standardize the matrix output
    #ss = StandardScaler()
    #count_matrix_scaled = ss.fit_transform(count_matrix) 
    
    # apply dimensionality reduction using PCA
    #count_matrix_pca = decomposition.PCA(n_components=2).fit_transform(count_matrix_scaled)
    
    indices = pd.Series(data['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
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
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)

    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(data['title'])[i])
    return recommended_movies
