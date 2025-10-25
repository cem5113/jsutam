import streamlit as st
from ui.home import render_home  # ui/home.py içinde var

st.set_page_config(page_title="🏠 Ana Sayfa", layout="wide")
render_home()
