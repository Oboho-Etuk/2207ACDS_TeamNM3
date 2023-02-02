# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Company Logo
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path='resources/imgs/Logo2_Infinity.png', width=300, height=300)
st.sidebar.image(my_logo)

# Page Title
st.title("Projects")

st.subheader("Infinity [One](http://34.241.15.160:5000/#movie-recommender-engine)")

st.subheader("Infinity Advantage")

st.subheader("Infinity Plus")

bottom_image = st.file_uploader('', type='png', key=6)
if bottom_image is not None:
    image = Image.open(bottom_image)
    new_image = 'resources/imgs/Logo2_Infinity.png'.resize((600, 400))
    st.image(new_image)
st.image('resources/imgs/Logo2_Infinity.png', caption='Infinity AI')

