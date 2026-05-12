import streamlit as st
import pandas as pd


def prediction_page():

    st.subheader("🔮 Churn Prediction")

    if "model" not in st.session_state:
        st.warning("Train models first")
        return

    model = st.session_state.model
    df = st.session_state.train_df

    input_data = {}

    cols = st.columns(2)

    features = df.columns[:-1]

    for i, col in enumerate(features):

        if df[col].dtype == 'object':

            val = cols[i % 2].selectbox(
                col,
                df[col].unique()
            )

        else:

            val = cols[i % 2].slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

        input_data[col] = val

    if st.button("Predict Churn"):

        input_df = pd.DataFrame([input_data])

        pred = model.predict(input_df)[0]

        if pred == 1:
            st.error("⚠️ High Churn Risk")
        else:
            st.success("✅ Low Churn Risk")