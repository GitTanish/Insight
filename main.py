import matplotlib

matplotlib.use("Agg")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Insight — AI Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import ui_components

ui_components.render_app()
