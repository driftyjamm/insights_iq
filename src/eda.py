import streamlit as st
import plotly.express as px


def run_eda(df):

    st.title("📊 Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Missing Values")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing"]

    fig = px.bar(
        missing,
        x="Column",
        y="Missing",
        color="Missing"
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:

        selected = st.selectbox(
            "Select Numeric Column",
            numeric_cols
        )

        fig2 = px.histogram(
            df,
            x=selected,
            nbins=30,
            title=f"Distribution of {selected}"
        )

        fig2.update_layout(template="plotly_dark")

        st.plotly_chart(fig2, use_container_width=True)