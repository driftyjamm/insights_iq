import streamlit as st
import pandas as pd

from src.preprocess import clean_data
from src.model import train_model
from src.prediction import prediction_page
from src.insights import show_insights

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="InsightsIQ",
    layout="wide",
    page_icon="📊"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #050814;
    color: white;
}

/* Logo */
.logo {
    font-size: 34px;
    font-weight: 800;
    color: white;
    padding-top: 10px;
}

/* Top nav */
.nav-container {
    display: flex;
    justify-content: center;
    gap: 18px;
    background: rgba(255,255,255,0.04);
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 30px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

/* Header */
.main-title {
    font-size: 48px;
    font-weight: 800;
    margin-top: 10px;
}

.sub-title {
    color: #8b9dc3;
    font-size: 18px;
    margin-bottom: 40px;
}

/* 3D cards */
.metric-card {
    background: linear-gradient(145deg, #111827, #1f2937);
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    box-shadow:
        8px 8px 20px rgba(0,0,0,0.55),
        -4px -4px 12px rgba(255,255,255,0.04);
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-title {
    color: #9ca3af;
    font-size: 16px;
}

.metric-value {
    font-size: 34px;
    font-weight: bold;
    color: #60a5fa;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="logo">INSIGHTSIQ</div>', unsafe_allow_html=True)

# ---------------- TOP NAV ----------------
tabs = [
    "Dashboard",
    "Upload Dataset",
    "Preprocessing",
    "Model Training",
    "Prediction",
    "Insights"
]

selected = st.radio(
    "",
    tabs,
    horizontal=True
)

st.markdown("---")

# ---------------- HERO ----------------
st.markdown('<div class="main-title">AI Customer Churn Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enterprise-grade predictive intelligence platform</div>', unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- DASHBOARD ----------------
if selected == "Dashboard":

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Models</div>
            <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Dataset Size</div>
            <div class="metric-value">Live</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Prediction Engine</div>
            <div class="metric-value">AI</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Status</div>
            <div class="metric-value">Active</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
elif selected == "Upload Dataset":

    st.header("Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.success("Dataset uploaded successfully")
        st.dataframe(df.head())

# ---------------- PREPROCESSING ----------------
elif selected == "Preprocessing":

    if st.session_state.df is not None:
        st.session_state.df = clean_data(st.session_state.df)
        st.success("Data cleaned successfully")
        st.dataframe(st.session_state.df.head())
    else:
        st.warning("Upload dataset first")

# ---------------- MODEL TRAINING ----------------
elif selected == "Model Training":

    if st.session_state.df is not None:
        train_model(st.session_state.df)
    else:
        st.warning("Upload dataset first")

# ---------------- PREDICTION ----------------
elif selected == "Prediction":

    prediction_page()

# ---------------- INSIGHTS ----------------
elif selected == "Insights":

    show_insights()
