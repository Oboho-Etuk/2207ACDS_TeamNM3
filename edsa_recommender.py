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
# install numpy and surprise packages
!pip install numpy
!pip install scikit-surprise

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
	st.info("Team Information - Experts and their Roles")
	# You can read a markdown file from supporting resources folder
	st.markdown("This Web app has been adapted and developed by the Infinity AI, a                              \
	leading data science firm with AI-powered solutions and its own R&D Center.                              \
	Infinity AI’s mission is to help our clients improve competitiveness and get great results from their work.          \
	We strive to reach these goals applying innovative and proprietary development technologies,             \
	providing exceptional services, and using excellent professional expertise.                            \n\
	We help businesses get valuable insights into data, automate repetitive tasks,                          \
	enhance performance, add AI-driven features, and prevent cost overruns.")
	
	st.subheader("Meet the Team")
	if st.button('Aniedi'): # information is hidden if button is clicked
		st.markdown('Aniedi Oboho-Etuk is a Infinity AI Developer')
	if st.button('Bongani'): # information is hidden if button is clicked
		st.markdown('Bongani Mavuso is the Infinity AI Project Manager')
	if st.button('Manoko'): # information is hidden if button is clicked
		st.markdown('Manoko Langa is an Infity AI Developer/Strategist')
	if st.button('Josiah'): # information is hidden if button is clicked
		st.markdown('Josiah Aramide is the Infinity AI CEO')
	if st.button('Tshepiso'): # information is hidden if button is clicked
		st.markdown('Tshepiso Padi is the Infinity AI Product Owner')
	if st.button('Ndinnanyi'): # information is hidden if button is clicked
		st.markdown('Ndinanyi Justice is the Infinity AI Brand Developer')
	
    	# DO NOT REMOVE the 'Recommender System' option below, however,
    	# you are welcome to add more options to enrich your app.
	page_options = ["Recommender System","Solution Overview"]
	
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
					top_recommendations = collab_model(movie_list=fav_movies, top_n=10)
					st.title("We think you'll like:")
					for i,j in enumerate(top_recommendations):
						st.subheader(str(i+1)+'. '+j)
			except:
				st.error("Oops! Looks like this algorithm does't work. \
				We'll need to fix it!")
				
	# -------------------------------------------------------------------
	
	# ------------- SAFE FOR ALTERING/EXTENSION -------------------
	if page_selection == "Solution Overview":
		st.title("Solution Overview")
		st.write("Describe your winning approach on this page")

    	# You may want to add more sections here for aspects such as an EDA,
    	# or to provide your business pitch.


if __name__ == '__main__':
    main()
