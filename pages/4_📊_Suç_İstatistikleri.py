import streamlit as st
from ui.tab_stats import render_stats  # ui/tab_stats.py

st.set_page_config(page_title="📊 Suç İstatistikleri", layout="wide")
render_stats()
