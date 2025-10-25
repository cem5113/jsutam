import streamlit as st
from ui.tab_planning import render_planning  # ui/tab_planning.py

st.set_page_config(page_title="🚓 Devriye Planlama", layout="wide")
render_planning()
