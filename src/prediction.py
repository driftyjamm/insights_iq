import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def prediction_page():

    st.title("Customer Churn Prediction Center")

    if "model" not in st.session_state:
        st.warning("Train models first")
        return

    model = st.session_state.model
    columns = st.session_state.columns
    df = st.session_state.original_df

    st.subheader("Customer Input Panel")

    input_data = {}

    cols = st.columns(2)

    feature_cols = [col for col in df.columns if col not in ["Churn", "Class"]]

    for i, col in enumerate(feature_cols[:8]):

        if pd.api.types.is_numeric_dtype(df[col]):

            val = cols[i % 2].slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean()),
                key=f"pred_{col}"
            )

        else:

            val = cols[i % 2].selectbox(
                col,
                df[col].dropna().unique(),
                key=f"pred_{col}"
            )

        input_data[col] = val

    if st.button("Generate Prediction"):

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        for col in columns:
            if col not in input_df:
                input_df[col] = 0

        input_df = input_df[columns]

        pred = model.predict(input_df)[0]

        st.markdown("---")

        if pred == 1:
            st.error("⚠ High Churn Risk")
            risk = 85
        else:
            st.success("✅ Low Churn Risk")
            risk = 20

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
            risk = prob * 100

        st.metric("Risk Probability", f"{risk:.2f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text': "Churn Risk Score"}
        ))

        fig.update_layout(
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("""
Prediction Meaning:

- 0–30% → Stable customer  
- 30–60% → Moderate retention concern  
- 60–100% → High churn probability
""")
