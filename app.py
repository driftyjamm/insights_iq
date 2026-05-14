import streamlit as st
import pandas as pd

from src.preprocessor import clean_data
from src.model import train_model
from src.model import model_comparison
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

/* Main app */
.stApp {
    background-color: #f8fafc;
    color: #1e293b;
}

/* Logo */
.logo {
    font-size: 34px;
    font-weight: 800;
    color: #111827;
    padding-top: 10px;
}

/* Navigation */
.nav-container {
    display: flex;
    justify-content: center;
    gap: 18px;
    background: #ffffff;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 30px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}

/* Main heading */
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #111827;
}

.sub-title {
    color: #64748b;
    font-size: 18px;
    margin-bottom: 40px;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    border: 1px solid #e5e7eb;
    box-shadow:
        0 10px 25px rgba(0,0,0,0.08),
        inset 0 1px 0 rgba(255,255,255,0.9);
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 16px 35px rgba(0,0,0,0.12);
}

.metric-title {
    color: #6b7280;
    font-size: 16px;
}

.metric-value {
    font-size: 34px;
    font-weight: bold;
    color: #2563eb;
}

/* Tables */
[data-testid="stDataFrame"] {
    background: white;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Buttons */
.stButton>button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
}

.stButton>button:hover {
    background: #1d4ed8;
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
    "Model Comparison",
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

elif selected == "Model Comparison":
    model_comparison()

# ---------------- PREDICTION ----------------
elif selected == "Prediction":

    prediction_page()

# ---------------- INSIGHTS ----------------
elif selected == "Insights":

    show_insights()
