import streamlit as st
import plotly.express as px


def show_comparison():

    st.subheader("📊 Model Comparison")

    if "results" not in st.session_state:
        st.warning("Train models first")
        return

    df = st.session_state.results

    st.dataframe(df)

    fig1 = px.bar(
        df,
        x="Model",
        y="Accuracy",
        color="Model",
        title="Accuracy Comparison"
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(
        df,
        x="Model",
        y=["Precision", "Recall", "F1 Score"],
        markers=True,
        title="Metrics Comparison"
    )

    st.plotly_chart(fig2, use_container_width=True)