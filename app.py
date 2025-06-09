import streamlit as st
import streamlit as st
from upload_page import show_upload_page
from tweet_page import show_tweet_page

# Sidebar for navigation

page = st.sidebar.selectbox("Main Menu", ("Upload Dataset", "Input Text"))

if page == "Upload Dataset":
   show_upload_page()
else:
   show_tweet_page()