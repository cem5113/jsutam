import streamlit as st
from ui.tab_reports import render_reports  # ui/tab_reports.py

st.set_page_config(page_title="🧾 Raporlar & Operasyonel Öneriler", layout="wide")
render_reports()
