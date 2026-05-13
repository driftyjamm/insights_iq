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

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)


# ---------------- CLEAN ----------------
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


# ---------------- TARGET VALIDATION ----------------
def prepare_target(y):

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    unique_vals = len(pd.Series(y).unique())

    if unique_vals > 10:
        st.error(
            "❌ Selected target is continuous.\n\n"
            "Choose a binary/categorical target like **Churn / Class / Fraud**"
        )
        return None

    return y


# ---------------- MODEL EXPLANATIONS ----------------
MODEL_INFO = {
    "Random Forest": """
Builds multiple decision trees using random subsets.
Final decision = majority voting.
""",

    "Gradient Boosting": """
Sequential trees correct previous errors.
Optimizes prediction step-by-step.
""",

    "Extra Trees": """
Highly randomized tree splits.
Fast and prevents overfitting.
""",

    "AdaBoost": """
Focuses on wrongly predicted samples.
Improves weak learners iteratively.
"""
}


# ---------------- CARD METRICS ----------------
def metric_cards(acc, prec, rec, f1):

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1 Score", f"{f1:.3f}")


# ---------------- DASHBOARD ----------------
def show_model_dashboard(name, model, X_test, y_test, preds, probs, X):

    st.markdown(f"# {name}")

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    metric_cards(acc, prec, rec, f1)

    st.markdown("---")

    # ---------------- 1. Confusion Matrix ----------------
    cm = confusion_matrix(y_test, preds)

    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["Actual 0", "Actual 1"]
    )

    fig_cm.update_layout(template="plotly_dark", height=260)

    st.plotly_chart(fig_cm, use_container_width=True, key=f"{name}_cm")

    st.info("""
**Confusion Matrix Explanation**

Shows how many predictions were correct vs incorrect.

- Top-left → Correct negative predictions  
- Bottom-right → Correct positive predictions  
- Other cells → Wrong predictions

A strong model has high diagonal values.
""")

    # ---------------- 2. ROC Curve ----------------
    if probs is not None and len(set(y_test)) == 2:

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr,
            y=tpr,
            title=f"ROC Curve (AUC={roc_auc:.2f})"
        )

        fig_roc.update_layout(template="plotly_dark", height=260)

        st.plotly_chart(fig_roc, use_container_width=True, key=f"{name}_roc")

        st.info(f"""
**ROC Curve Explanation**

Measures classification quality.

- X-axis → False Positive Rate  
- Y-axis → True Positive Rate  
- AUC = {roc_auc:.2f}

Higher AUC means better separation between churn and non-churn customers.
""")

    # ---------------- 3. Feature Importance ----------------
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

        fig_feat.update_layout(template="plotly_dark", height=280)

        st.plotly_chart(fig_feat, use_container_width=True, key=f"{name}_feat")

        st.info("""
**Feature Importance Explanation**

Ranks the variables affecting prediction most.

Higher bars indicate stronger influence on churn prediction.

This helps identify which customer attributes drive churn behavior.
""")

    # ---------------- 4. Prediction Distribution ----------------
    fig_pred = px.histogram(
        x=preds,
        title="Prediction Distribution"
    )

    fig_pred.update_layout(template="plotly_dark", height=250)

    st.plotly_chart(fig_pred, use_container_width=True, key=f"{name}_pred")

    st.info("""
**Prediction Distribution Explanation**

Shows how many customers were predicted in each class.

Balanced distribution indicates stable model behavior.
""")

    # ---------------- 5. Metric Trend ----------------
    metric_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Score": [acc, prec, rec, f1]
    })

    fig_metric = px.line(
        metric_df,
        x="Metric",
        y="Score",
        markers=True
    )

    fig_metric.update_layout(template="plotly_dark", height=250)

    st.plotly_chart(fig_metric, use_container_width=True, key=f"{name}_metric")

    st.info("""
**Metric Trend Explanation**

Compares all evaluation metrics.

- Accuracy → Overall correctness  
- Precision → Reliability of positive predictions  
- Recall → Ability to detect churn cases  
- F1 Score → Balance between precision and recall
""")

    # ---------------- 6. Gauge ----------------
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=acc * 100,
        title={'text': "Model Accuracy"}
    ))

    fig_gauge.update_layout(template="plotly_dark", height=250)

    st.plotly_chart(fig_gauge, use_container_width=True, key=f"{name}_gauge")

    st.info("""
**Accuracy Gauge Explanation**

Provides an instant performance summary.

Closer to 100% means stronger prediction capability.
""")

    # ---------------- Algorithm ----------------
    st.markdown("## Algorithm Working")
    st.info(MODEL_INFO[name])

# ---------------- MAIN TRAIN ----------------
def train_model(df):

    st.title("Advanced Model Training")

    df = clean_data(df)

    target = st.selectbox("Select Target Column", df.columns)

    X = pd.get_dummies(df.drop(target, axis=1))
    y = prepare_target(df[target])

    if y is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    if st.button("Train Models"):

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
