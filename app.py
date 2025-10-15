# app.py (minimal giriş)
import streamlit as st

st.set_page_config(page_title="Suç Tahmini Uygulaması", layout="wide")
st.title("🗺️ Suç Tahmini Uygulaması")

st.write("Soldaki **Pages** menüsünden sayfalara gidebilirsiniz.")
try:
    st.page_link("pages/Home.py", label="→ Günlük Risk Haritası", icon="🗺️")
    st.page_link("pages/Forecast.py", label="→ Forecast (Model+)", icon="🔮")
except Exception:
    pass
