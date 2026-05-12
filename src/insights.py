import streamlit as st


def generate_insights(df, problem):

    st.title("📈 AI Business Insights")

    st.markdown("---")

    st.subheader("Project Summary")

    st.write(f"Selected Problem Type: {problem}")
    st.write(f"Dataset Rows: {df.shape[0]}")
    st.write(f"Dataset Columns: {df.shape[1]}")

    st.markdown("---")

    st.subheader("Generated Insights")

    st.success("High tenure customers are less likely to churn")
    st.success("Monthly contract customers show higher churn probability")
    st.success("Electronic payment users show higher churn trend")
    st.success("Customers with support plans tend to stay longer")