import streamlit as st
import pandas as pd

from src.preprocessor import clean_data
from src.model import train_model
from src.comparison import show_comparison
from src.prediction import prediction_page
from src.insights import generate_insights

st.set_page_config(
    page_title="InsightsIQ",
    page_icon="🧠",
    layout="wide"
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.sidebar:

    st.title("🧠 InsightsIQ")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Upload Dataset",
            "Preprocessing",
            "Model Training",
            "Model Comparison",
            "Prediction",
            "Insights"
        ]
    )

if "df" not in st.session_state:
    st.session_state.df = None

st.markdown("<div class='main-title'>InsightsIQ</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Powered Customer Churn Analytics Platform</div>", unsafe_allow_html=True)

if page == "Dashboard":

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Models", "4")
    col2.metric("AI Type", "Ensemble")
    col3.metric("Dataset", "Telecom")
    col4.metric("Status", "Ready")

    st.markdown("---")

    st.info("Upload telecom churn dataset to begin training advanced AI models.")

elif page == "Upload Dataset":

    st.subheader("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:

        df = pd.read_csv(file)
        st.session_state.df = df

        st.success("Dataset uploaded successfully")

        st.dataframe(df.head())

elif page == "Preprocessing":

    st.subheader("🧹 Data Cleaning & Preprocessing")

    if st.session_state.df is not None:

        cleaned_df = clean_data(st.session_state.df)
        st.session_state.cleaned_df = cleaned_df

        st.success("Preprocessing completed")

        st.dataframe(cleaned_df.head())

    else:
        st.warning("Upload dataset first")

elif page == "Model Training":

    if "cleaned_df" in st.session_state:

        train_model(st.session_state.cleaned_df)

    else:
        st.warning("Complete preprocessing first")

elif page == "Model Comparison":

    show_comparison()

elif page == "Prediction":

    prediction_page()

elif page == "Insights":

    if "cleaned_df" in st.session_state:
        generate_insights(st.session_state.cleaned_df)
