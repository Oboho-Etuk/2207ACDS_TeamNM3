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
st.image(my_logo)
st.sidebar.image(my_logo)

# contact us form field
st.title("Contact Us")
with st.form('form1', clear_on_submit=True):
    name = st.text_input("Enter your full name")
    address = st.text_input("Type in your postcode")
    email = st.text_input("Enter email")
    message = st.text_input("Type in your enquiry or message")
    age = st.slider("Enter your age", min_value=10, max_value=80)
    st.write(age)

    submit = st.form_submit_button("Submit this form")


st.subheader("Team Leader/CEO")
st.caption("Kaggle Username: [Kaggle](https://www.kaggle.com/bonganimavuso)")
st.caption("Trello Username: [Trello](https://trello.com/u/bonganimavuso1)")
st.caption("Github Username: BMavuso")
st.caption("Email address: [Gmail](mavusobss@gmail.com)")
st.caption("Canva Username: Bongani_Mavuso")