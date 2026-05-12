import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)


# ---------------- CLEAN DATA ----------------
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


# ---------------- MODEL DETAILS ----------------
MODEL_INFO = {
    "Random Forest": {
        "algo": """
Random Forest builds multiple decision trees using random subsets of data
and combines them through majority voting.

Working:
1. Bootstrap sampling
2. Build many trees
3. Each tree predicts
4. Final output via voting
"""
    },

    "Gradient Boosting": {
        "algo": """
Gradient Boosting builds trees sequentially.

Each new tree corrects errors of previous trees.
Optimizes prediction through gradient descent.
"""
    },

    "Extra Trees": {
        "algo": """
Extra Trees randomly selects split points.

Creates highly randomized trees for faster training
and better generalization.
"""
    },

    "AdaBoost": {
        "algo": """
AdaBoost focuses on incorrectly predicted samples.

Each weak learner improves previous mistakes.
Combines them into a strong learner.
"""
    }
}


# ---------------- SHOW MODEL DASHBOARD ----------------
def show_model_dashboard(name, model, X_test, y_test, preds, probs, X):

    st.subheader(f"📌 {name}")

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Precision", f"{prec:.2f}")
    c3.metric("Recall", f"{rec:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")

    st.markdown("---")

    # ---------------- GRAPH 1: Confusion Matrix ----------------
    cm = confusion_matrix(y_test, preds)

    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=["No Churn", "Churn"],
        y=["No Churn", "Churn"]
    )

    fig_cm.update_layout(title="Confusion Matrix", template="plotly_dark")
    st.plotly_chart(fig_cm, use_container_width=True, key=f"{name}_cm")

    # ---------------- GRAPH 2: ROC Curve ----------------
    if probs is not None:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.2f}"))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark"
        )

        st.plotly_chart(fig_roc, use_container_width=True, key=f"{name}_roc")

    # ---------------- GRAPH 3: Feature Importance ----------------
    if hasattr(model, "feature_importances_"):

        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)

        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Features"
        )

        fig_feat.update_layout(template="plotly_dark")
        st.plotly_chart(fig_feat, use_container_width=True, key=f"{name}_feature")

    # ---------------- GRAPH 4: Prediction Distribution ----------------
    pred_df = pd.DataFrame({"Prediction": preds})

    fig_pred = px.histogram(
        pred_df,
        x="Prediction",
        title="Prediction Distribution"
    )

    fig_pred.update_layout(template="plotly_dark")
    st.plotly_chart(fig_pred, use_container_width=True, key=f"{name}_prediction")

    # ---------------- GRAPH 5: Metrics Comparison ----------------
    metric_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Score": [acc, prec, rec, f1]
    })

    fig_metrics = px.line(
        metric_df,
        x="Metric",
        y="Score",
        markers=True,
        title="Metric Trend"
    )

    fig_metrics.update_layout(template="plotly_dark")
    st.plotly_chart(fig_metrics, use_container_width=True, key=f"{name}_metrics")

    # ---------------- GRAPH 6: Gauge ----------------
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=acc * 100,
        title={'text': "Model Accuracy"}
    ))

    fig_gauge.update_layout(template="plotly_dark")
    st.plotly_chart(fig_gauge, use_container_width=True, key=f"{name}_gauge")

    # ---------------- ALGORITHM EXPLANATION ----------------
    st.markdown("## 🧠 Algorithm Working")
    st.info(MODEL_INFO[name]["algo"])

    st.markdown("## 🔄 Workflow")

    st.code("""
Input Dataset
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Prediction
   ↓
Evaluation
""")


# ---------------- MAIN TRAIN FUNCTION ----------------
def train_model(df):

    st.title("🧠 Advanced Model Training")

    df = clean_data(df)

    target = st.selectbox("Select Target Column", df.columns)

    X = pd.get_dummies(df.drop(target, axis=1))
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("🚀 Train Models"):

        models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }

        trained = {}

        for name, model in models.items():
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]

            trained[name] = {
                "model": model,
                "preds": preds,
                "probs": probs
            }

        st.success("All models trained successfully!")

        tabs = st.tabs(list(models.keys()))

        for i, name in enumerate(models.keys()):
            with tabs[i]:
                show_model_dashboard(
                    name,
                    trained[name]["model"],
                    X_test,
                    y_test,
                    trained[name]["preds"],
                    trained[name]["probs"],
                    X
                )

        # Save best model
        st.session_state.model = trained["Random Forest"]["model"]
        st.session_state.columns = X.columns
        st.session_state.original_df = df
