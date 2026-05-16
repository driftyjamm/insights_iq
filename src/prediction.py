import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def prediction_page():

    st.title("Customer Churn Prediction Center")
    st.markdown("Enterprise AI-powered churn risk assessment engine")

    if "model" not in st.session_state:
        st.warning("Train models first")
        return

    model = st.session_state.model
    columns = st.session_state.columns
    df = st.session_state.original_df

    # ---------------- HOW IT WORKS ----------------
    st.subheader("How Prediction Works")

    st.info("""
**Prediction Workflow**

1. Raw customer data is collected  
2. Data is cleaned and preprocessed  
3. Features are transformed into machine-readable format  
4. The trained AI model identifies hidden churn patterns  
5. Risk probability is calculated  
6. Final churn prediction is generated
""")

    # ---------------- FEATURES USED ----------------
    st.subheader("Features Used By Model")

    feature_df = pd.DataFrame({
        "Feature Used for Prediction": columns
    })

    st.dataframe(feature_df, use_container_width=True)

    st.success(f"""
The model is currently using **{len(columns)} engineered features**
to detect customer churn patterns.
""")

    st.markdown("---")

    # ---------------- INPUT PANEL ----------------
    st.subheader("Customer Input Panel")

    input_data = {}

    cols = st.columns(2)

    feature_cols = [col for col in df.columns if col not in ["Churn", "Class"]]

    for i, col in enumerate(feature_cols[:8]):

        if pd.api.types.is_numeric_dtype(df[col]):

            # Clean numeric values
            col_data = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(col_data) == 0:
                continue

            min_val = float(col_data.min())
            max_val = float(col_data.max())
            default_val = float(col_data.mean())

            # Prevent slider crash if min == max
            if min_val == max_val:
                max_val = min_val + 1

            val = cols[i % 2].slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"pred_{col}"
            )

        else:

            val = cols[i % 2].selectbox(
                col,
                df[col].dropna().unique(),
                key=f"pred_{col}"
            )

        input_data[col] = val

    # ---------------- PREDICTION ----------------
    if st.button("Generate Prediction"):

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        for col in columns:
            if col not in input_df:
                input_df[col] = 0

        input_df = input_df[columns]

        pred = model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        if pred == 1:
            st.error("⚠ High Churn Risk")
            risk = 85
            explanation = """
This customer shows a strong likelihood of leaving.

Recommended Actions:
- Immediate retention outreach
- Personalized discount offer
- Customer satisfaction review
"""
        else:
            st.success("✅ Low Churn Risk")
            risk = 20
            explanation = """
This customer appears stable.

Recommended Actions:
- Maintain engagement
- Continue standard support
- Monitor periodically
"""

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
            risk = prob * 100

        # ---------------- METRIC ----------------
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Risk Probability", f"{risk:.2f}%")

        with c2:
            if risk > 60:
                st.metric("Retention Priority", "High")
            elif risk > 30:
                st.metric("Retention Priority", "Medium")
            else:
                st.metric("Retention Priority", "Low")

        # ---------------- GAUGE ----------------
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

        # ---------------- INTERPRETATION ----------------
        st.info(f"""
**Prediction Interpretation**

- **0–30%** → Stable customer  
- **30–60%** → Moderate churn possibility  
- **60–100%** → High churn probability  

Current Result: **{risk:.2f}%**
""")

        # ---------------- BUSINESS INSIGHT ----------------
        st.subheader("Business Recommendation")
        st.write(explanation)
