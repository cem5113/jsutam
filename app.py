# app.py — SUTAM (revize tam sürüm)
import streamlit as st
from components.last_update import show_last_update_badge
from components.utils.constants import MODEL_VERSION, MODEL_LAST_TRAIN

st.set_page_config(page_title="Suç Tahmini Uygulaması", layout="wide")
st.title("Suç Tahmini Uygulaması")
st.write("Soldaki **Pages** menüsünden sekmelere geçebilirsiniz.")
st.info("🔎 Harita için: **🧭 Suç Tahmini** sekmesine gidin.")

# (Opsiyonel) model rozeti
show_last_update_badge(
    data_upto=None,              # veri tarihi göstermek istemezsen None bırak
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)
