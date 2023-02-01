# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

st.title("Projects")

st.subheader("Infinity One")

st.subheader("Infinity Advantage")

st.subheader("Infinity Plus")

bottom_image = st.file_uploader('', type='png', key=6)
if bottom_image is not None:
    image = Image.open(bottom_image)
    new_image = 'resources/imgs/Logo2_Infinity.png'.resize((600, 400))
    st.image(new_image)
st.image('resources/imgs/Logo2_Infinity.png', caption='Infinity AI')

