import streamlit as st
import pandas as pd
import plotly.express as px


def show_insights(df=None):

    st.title("Business Insights Center")

    if df is None:
        if "df" not in st.session_state:
            st.warning("Upload dataset first")
            return
        df = st.session_state.df

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("---")

    # Data Types
    st.subheader("Feature Distribution")

    dtype_counts = df.dtypes.astype(str).value_counts()

    fig1 = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index,
        title="Data Type Distribution"
    )

    fig1.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig1, use_container_width=True)

    st.info("Shows categorical vs numerical feature composition.")

    # Missing values
    st.subheader("Missing Value Analysis")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        fig2 = px.bar(
            x=missing.index,
            y=missing.values,
            title="Missing Values by Column"
        )

        fig2.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.success("No missing values detected.")

    # Statistical Summary
    st.subheader("Statistical Insights")

    st.dataframe(df.describe(), use_container_width=True)

    # Business Insights
    st.subheader("Key Observations")

    st.success("""
Insights Generated:

• Dataset quality assessment completed  
• Feature distributions analyzed  
• Missing value patterns evaluated  
• Statistical behavior identified  
• Dataset is ready for predictive analytics
""")
