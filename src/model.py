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


# ---------------- TARGET VALIDATION ----------------
def prepare_target(y):

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    unique_vals = len(pd.Series(y).unique())

    if unique_vals > 10:
        st.error(
            "❌ Invalid target column.\n\n"
            "Select categorical/binary target like Churn / Class"
        )
        return None

    return y


# ---------------- MODEL INFO ----------------
MODEL_INFO = {
    "Random Forest": "Builds multiple decision trees and combines them using majority voting.",
    "Gradient Boosting": "Sequentially corrects errors from previous trees.",
    "Extra Trees": "Highly randomized tree ensemble for fast generalization.",
    "AdaBoost": "Boosts weak learners by focusing on misclassified samples."
}


# ---------------- METRIC CARDS ----------------
def metric_cards(acc, prec, rec, f1):

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1 Score", f"{f1:.3f}")


# ---------------- MODEL DASHBOARD ----------------
def show_model_dashboard(name, model, X_test, y_test, preds, probs, X):

    st.header(name)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    metric_cards(acc, prec, rec, f1)

    st.markdown("---")

    # 1 Confusion Matrix
    cm = confusion_matrix(y_test, preds)

    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["Actual 0", "Actual 1"]
    )
    fig_cm.update_layout(template="plotly_dark", height=260)

    st.plotly_chart(fig_cm, use_container_width=True, key=f"{name}_cm")
    st.info("Shows correct vs incorrect predictions.")

    # 2 ROC
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
        st.info("Higher AUC means stronger classification ability.")

    # 3 Feature Importance
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
        st.info("Ranks most influential features.")

    # 4 Prediction Distribution
    fig_pred = px.histogram(x=preds, title="Prediction Distribution")
    fig_pred.update_layout(template="plotly_dark", height=250)

    st.plotly_chart(fig_pred, use_container_width=True, key=f"{name}_pred")
    st.info("Shows predicted class balance.")

    # 5 Metric Trend
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
    st.info("Compares all evaluation metrics.")

    # 6 Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=acc * 100,
        title={'text': "Accuracy"}
    ))

    fig_gauge.update_layout(template="plotly_dark", height=250)

    st.plotly_chart(fig_gauge, use_container_width=True, key=f"{name}_gauge")
    st.info("Overall model performance score.")

    st.subheader("Algorithm Working")
    st.write(MODEL_INFO[name])


# ---------------- TRAIN MODELS ----------------
def train_model(df):

    st.title("Advanced Model Training")

    target = st.selectbox("Select Target Column", df.columns)

    X = pd.get_dummies(df.drop(target, axis=1))
    y = prepare_target(df[target])

    if y is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("Train Models"):

        models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }

        trained = {}
        results = []

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")
            f1 = f1_score(y_test, preds, average="weighted")

            trained[name] = {
                "model": model,
                "preds": preds,
                "probs": probs
            }

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })

        st.session_state.model_results = pd.DataFrame(results)

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

        best_model = max(
            trained.items(),
            key=lambda x: accuracy_score(y_test, x[1]["preds"])
        )

        st.session_state.model = best_model[1]["model"]
        st.session_state.columns = X.columns
        st.session_state.original_df = df


# ---------------- COMPARISON PAGE ----------------
def model_comparison():

    if "model_results" not in st.session_state:
        st.warning("Train models first")
        return

    st.title("Model Comparison")

    results = st.session_state.model_results

    st.dataframe(results, use_container_width=True)

    fig1 = px.bar(
        results,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        title="Accuracy Comparison"
    )

    fig1.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig1, use_container_width=True)

    metric_data = results.melt(
        id_vars=["Model"],
        value_vars=["Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score"
    )

    fig2 = px.line(
        metric_data,
        x="Metric",
        y="Score",
        color="Model",
        markers=True
    )

    fig2.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    best = results.sort_values("Accuracy", ascending=False).iloc[0]

    st.success(
        f"Best Model: {best['Model']} "
        f"(Accuracy: {best['Accuracy']:.3f})"
    )
