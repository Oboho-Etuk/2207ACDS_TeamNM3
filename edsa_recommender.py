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
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "About The Team"]

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
	
	# Select Movies
	movie_list = movies['title'].values
	
	selected_movie = st.selectbox( "Type or select a movie from the dropdown", movie_list )
	
	if st.button('Show Recommendation'):
		recommended_movie_names = get_recommendations(selected_movie)
		recommended_movie_names
		
		
	# Plotly Table
	def table(df):
		fig=go.Figure(go.table( columnorder = [1,2,3],
				       columnwidth = [10,28],
				       header=dict(values=[' title','genres'],
						   line_color='black',font=dict(color='black',size= 19),height=40,
						   fill_color='#dd571c',#
						   align=['left','center']),
				       cells=dict(values=[movies.title,movies.genres],
						  fill_color='#ffdac4',line_color='grey',
						  font=dict(color='black', family="Lato", size=16),
						  align='left')))
		fig.update_layout(height=600, title ={'text': "Top 10 Movie Recommendations", 'font': {'size': 22}},title_x=0.5)
		return fig.show()
	
	if st.button('Show Recommendation'):
		recommended_movie_names = get_recommendations(selected_movie)
		table(recommended_movie_names)
	
	
	# You may want to add more sections here for aspects such as an EDA,
    	# or to provide your business pitch.


if __name__ == '__main__':
    main()
