import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

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


# ---------------- TARGET ----------------
def prepare_target(y):

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    if len(pd.Series(y).unique()) > 10:
        st.error("Select categorical target like Churn/Class")
        return None

    return y


# ---------------- MODEL INFO ----------------
MODEL_INFO = {
    "Random Forest": "Multiple decision trees combined through majority voting.",
    "Gradient Boosting": "Sequentially corrects prediction errors.",
    "Extra Trees": "Randomized ensemble trees for faster learning.",
    "AdaBoost": "Boosts weak learners iteratively.",
    "XGBoost": "Optimized gradient boosting with regularization.",
    "CatBoost": "Handles categorical features efficiently."
}


# ---------------- DASHBOARD ----------------
def show_model_dashboard(name, model, X_test, y_test, preds, probs, X):

    st.header(name)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1", f"{f1:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)

    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["Actual 0", "Actual 1"]
    )

    fig_cm.update_layout(template="plotly_dark", height=260)
    st.plotly_chart(fig_cm, use_container_width=True, key=f"{name}_cm")
    st.info("Prediction correctness distribution.")

    # ROC
    if probs is not None and len(set(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr,
            y=tpr,
            title=f"ROC AUC = {roc_auc:.3f}"
        )

        fig_roc.update_layout(template="plotly_dark", height=260)
        st.plotly_chart(fig_roc, use_container_width=True, key=f"{name}_roc")

    # Feature Importance
    if hasattr(model, "feature_importances_"):

        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)

        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        fig_feat.update_layout(template="plotly_dark", height=280)

        st.plotly_chart(fig_feat, use_container_width=True, key=f"{name}_feat")

    st.subheader("Algorithm Working")
    st.write(MODEL_INFO[name])


# ---------------- TRAIN ----------------
def train_model(df):

    st.title("Advanced Model Training")

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

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    st.success("SMOTE balancing applied")

    if st.button("Train Models"):

        rf_grid = GridSearchCV(
            RandomForestClassifier(),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10]
            },
            cv=3
        )

        rf_grid.fit(X_train, y_train)

        st.info(f"Optimized RF Params: {rf_grid.best_params_}")

        models = {
            "Random Forest": rf_grid.best_estimator_,
            "Gradient Boosting": GradientBoostingClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "XGBoost": XGBClassifier(eval_metric="logloss"),
            "CatBoost": CatBoostClassifier(verbose=0)
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

        best = max(
            trained.items(),
            key=lambda x: accuracy_score(y_test, x[1]["preds"])
        )

        st.session_state.model = best[1]["model"]
        st.session_state.columns = X.columns
        st.session_state.original_df = df


# ---------------- COMPARISON ----------------
def model_comparison():

    if "model_results" not in st.session_state:
        st.warning("Train models first")
        return

    results = st.session_state.model_results.sort_values(
        "Accuracy",
        ascending=False
    )

    st.title("AI Model Benchmarking Center")

    # ---------------- LEADERBOARD CARDS ----------------
    st.subheader("Performance Leaderboard")

    cols = st.columns(len(results))
    medals = ["🥇", "🥈", "🥉", "⭐", "⚡", "🚀"]

    for i, (_, row) in enumerate(results.iterrows()):
    with cols[i]:
        st.markdown(f"""
        <div style="
            background: linear-gradient(145deg,#ffffff,#f1f5f9);
            padding:24px;
            border-radius:20px;
            border:1px solid #e5e7eb;
            box-shadow:0 12px 28px rgba(0,0,0,0.08);
            text-align:center;
            min-height:370px;
        ">
            <h3 style="color:#111827;">
                {medals[i]} {row['Model']}
            </h3>

            <h1 style="
                color:#2563eb;
                font-size:52px;
                margin:20px 0;
            ">
                {row['Accuracy']:.3f}
            </h1>

            <p style="color:#64748b;">Accuracy</p>

            <hr style="border:1px solid #e5e7eb;">

            <p style="color:#334155;">Precision: {row['Precision']:.3f}</p>
            <p style="color:#334155;">Recall: {row['Recall']:.3f}</p>
            <p style="color:#334155;">F1 Score: {row['F1 Score']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- HEATMAP ----------------
    st.subheader("Metric Heatmap")

    heat_df = results.set_index("Model")

    fig_heat = px.imshow(
        heat_df,
        text_auto=True,
        aspect="auto",
        title="Performance Matrix"
    )

    fig_heat.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.info(
        "This matrix highlights performance intensity across all evaluation metrics."
    )

    # ---------------- RADAR CHART ----------------
    st.subheader("Multi-Metric Radar Analysis")

    import plotly.graph_objects as go

    fig_radar = go.Figure()

    for _, row in results.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                row["Accuracy"],
                row["Precision"],
                row["Recall"],
                row["F1 Score"]
            ],
            theta=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1"
            ],
            fill='toself',
            name=row["Model"]
        ))

    fig_radar.update_layout(
        template="plotly_white",
        height=520
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    st.info(
        "Radar visualization helps compare model balance across all dimensions."
    )

    # ---------------- ACCURACY BAR ----------------
    st.subheader("Accuracy Benchmark")

    fig_bar = px.bar(
        results,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        text="Accuracy"
    )

    fig_bar.update_layout(
        template="plotly_white",
        height=420
    )

    fig_bar.update_traces(texttemplate="%{text:.3f}")

    st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- FINAL RECOMMENDATION ----------------
    best = results.iloc[0]

    st.success(f"""
Recommended Production Model: **{best['Model']}**

Why this model was selected:
- Highest predictive accuracy
- Strong metric consistency
- Reliable churn classification performance
- Best deployment readiness for enterprise analytics
""")
