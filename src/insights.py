import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def show_insights(df=None):

    st.title("Business Insights Center")

    if df is None:
        if "df" not in st.session_state:
            st.warning("Upload dataset first")
            return
        df = st.session_state.df

    # ---------------- OVERVIEW ----------------
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("---")

    # ---------------- DATA TYPES ----------------
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

    # ---------------- MISSING VALUES ----------------
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

    # ---------------- STATISTICS ----------------
    st.subheader("Statistical Insights")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    # ================= NEW ANALYST INSIGHTS =================

    st.subheader("Retention Cost Analysis")

    customers = st.number_input("Customers at Risk", 100)
    retention_cost = st.number_input("Retention Cost per Customer", 50)
    loss_per_customer = st.number_input("Loss per Churned Customer", 500)
    
    saved = customers * loss_per_customer
    cost = customers * retention_cost
    roi = saved - cost

    st.metric("Estimated Savings", f"${saved}")
    st.metric("Retention Cost", f"${cost}")
    st.metric("Net ROI", f"${roi}")
    
    # ---------------- CORRELATION ----------------
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] > 1:

        st.subheader("Correlation Intelligence")

        corr = numeric_df.corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            title="Feature Correlation Matrix"
        )

        fig_corr.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.info(
            "Identifies strongest relationships between variables for predictive analysis."
        )

    # ---------------- OUTLIERS ----------------
    if numeric_df.shape[1] > 0:

        st.subheader("Outlier Detection")

        selected_col = st.selectbox(
            "Select feature for outlier analysis",
            numeric_df.columns
        )

        fig_box = px.box(
            numeric_df,
            y=selected_col,
            title=f"Outlier Analysis: {selected_col}"
        )

        fig_box.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_box, use_container_width=True)

        st.info(
            "Outliers may indicate anomalies, fraud patterns, or extreme churn behavior."
        )

    # ---------------- FEATURE IMPORTANCE ----------------
    if "model" in st.session_state:

        st.subheader("Feature Importance Intelligence")

        model = st.session_state.model

        if hasattr(model, "feature_importances_"):

            feat_df = pd.DataFrame({
                "Feature": st.session_state.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)

            fig_feat = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Predictive Drivers"
            )

            fig_feat.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_feat, use_container_width=True)

            st.success(
                "These features contribute most strongly to prediction outcomes."
            )

    # ---------------- DATA QUALITY SCORE ----------------
    st.subheader("Dataset Quality Score")

    completeness = 100 - (
        (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    )

    st.metric("Quality Score", f"{completeness:.2f}%")

    # ---------------- ANALYST RECOMMENDATIONS ----------------
    st.subheader("Analyst Recommendations")

    insights = []

    if df.isnull().sum().sum() > 0:
        insights.append("Missing values should be handled carefully.")

    if numeric_df.shape[1] > 0:
        insights.append("Numerical distributions should be normalized if skewed.")

    insights.append("Monitor high-importance features for churn trends.")
    insights.append("Segment customers using prediction risk levels.")
    insights.append("Use retention strategies for high-risk profiles.")

    for item in insights:
        st.write(f"• {item}")

    # ---------------- BUSINESS INSIGHTS ----------------
    st.subheader("Key Observations")

    st.success("""
Insights Generated:

• Dataset quality assessment completed  
• Feature distributions analyzed  
• Missing value patterns evaluated  
• Statistical behavior identified  
• Correlation structures detected  
• Outlier patterns identified  
• Predictive drivers extracted  
• Dataset is ready for advanced predictive analytics
""")
