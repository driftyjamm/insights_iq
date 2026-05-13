import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans


def customer_segmentation(df):

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numeric features")
        return

    kmeans = KMeans(n_clusters=3, random_state=42)

    df["Cluster"] = kmeans.fit_predict(numeric_df)

    fig = px.scatter(
        df,
        x=numeric_df.columns[0],
        y=numeric_df.columns[1],
        color="Cluster",
        title="Customer Segmentation"
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)
