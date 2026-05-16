import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV
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
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve
        # ROC Curve
    if probs is not None and len(pd.Series(y_test).unique()) == 2:

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr,
            y=tpr,
            title=f"ROC Curve Analysis (AUC = {roc_auc:.3f})"
        )

        fig_roc.update_layout(
            template="plotly_dark",
            height=320,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        st.plotly_chart(fig_roc, use_container_width=True)

    # Professional Interpretation
        if roc_auc >= 0.90:
            performance = "Excellent"
            explanation = "The model has outstanding discrimination capability."
        elif roc_auc >= 0.80:
            performance = "Very Good"
            explanation = "The model separates churn and non-churn effectively."
        elif roc_auc >= 0.70:
            performance = "Good"
            explanation = "The model performs reasonably well with acceptable predictive capability."
        elif roc_auc >= 0.60:
            performance = "Moderate"
            explanation = "The model has some predictive power but needs improvement."
        else:
            performance = "Weak"
            explanation = "The model struggles to distinguish classes accurately."

        st.info(f"""
    ### ROC Curve Interpretation

    **AUC Score:** {roc_auc:.3f}

    **Performance Level:** {performance}

    **What this means:**  
    The ROC curve evaluates how effectively the model distinguishes between churned and retained customers.

    - Higher curve = Better classification
    - Top-left corner = Strong performance
    - AUC near 1.0 = Excellent
    - AUC near 0.5 = Random guessing

    **Interpretation:**  
    {explanation}
    """)
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
        st.plotly_chart(fig_feat, use_container_width=True)

    st.subheader("Algorithm Workflow")

    ALGO_DIAGRAMS = {

        "Random Forest": """
    Input Data
       ↓
    Bootstrap Sampling
       ↓
    Multiple Decision Trees
       ↓
    Independent Predictions
       ↓
    Majority Voting
       ↓
    Final Churn Prediction
    """,

        "Gradient Boosting": """
    Input Data
       ↓
    Train Weak Learner
       ↓
    Calculate Errors
       ↓
    Correct Residuals
       ↓
    Sequential Learning
       ↓
    Final Optimized Prediction
    """,

        "Extra Trees": """
    Input Data
       ↓
    Random Feature Selection
       ↓
    Random Split Generation
       ↓
    Multiple Random Trees
       ↓
    Aggregate Outputs
       ↓
    Final Prediction
    """,

        "AdaBoost": """
    Input Data
       ↓
    Train Weak Classifier
       ↓
    Increase Error Weights
       ↓
    Train Next Learner
       ↓
    Weighted Voting
       ↓
    Final Prediction
    """,

        "XGBoost": """
    Input Data
       ↓
    Gradient Optimization
       ↓
    Tree Construction
       ↓
    Regularization
       ↓
    Error Minimization
       ↓
    Final Prediction
    """,

        "CatBoost": """
    Input Data
       ↓
    Categorical Encoding
       ↓
    Ordered Boosting
       ↓
    Gradient Updates
       ↓
    Bias Reduction
       ↓
    Final Prediction
    """
    }

    st.code(ALGO_DIAGRAMS[name], language="text")

    st.subheader("Performance Interpretation")

    if acc >= 0.90:
        insight = "Exceptional predictive performance with strong class separation."
    elif acc >= 0.80:
        insight = "Strong predictive capability with reliable churn detection."
    elif acc >= 0.70:
        insight = """
    Moderate performance.

    This accuracy suggests the dataset contains partially separable churn patterns.

    A score around 0.70 is common in churn prediction because customer behavior is influenced by hidden business factors not fully captured in structured tabular data.

    Similar scores across models indicate:
    • Limited feature separability  
    • Dataset complexity  
    • Similar decision boundaries learned by ensemble models
    """
    elif acc >= 0.60:
        insight = "Model captures weak churn patterns and requires feature engineering."
    else:
        insight = "Low predictive capability. More preprocessing and better features required."

    st.info(insight)


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

    # SMOTE balancing
    # ---------------- SMART SMOTE ----------------
    class_counts = pd.Series(y_train).value_counts()

    if len(class_counts) > 1 and class_counts.min() >= 6:

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        st.success("SMOTE balancing applied")

    else:
        st.warning("""
    SMOTE skipped.

    Reason:
    Dataset has insufficient minority class samples.

    Model training will continue without balancing.
    """)

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

        num_classes = len(pd.Series(y_train).unique())

        models = {
            "Random Forest": rf_grid.best_estimator_,

            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300,
                max_depth=15
            ),

            "AdaBoost": AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.8
            ),

            "CatBoost": CatBoostClassifier(
                iterations=250,
                depth=6,
                verbose=0
            )
        }

        # Binary classification models
        if num_classes == 2:

            models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05
        )

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss"
        )

        # Multi-class support
        else:

            models["XGBoost"] = XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                objective="multi:softprob",
                num_class=num_classes
        )
            
        trained = {}
        results = []

        for name, model in models.items():

            try:
                model.fit(X_train, y_train)

            except Exception as e:
                st.warning(f"{name} skipped: {e}")
                continue

            preds = model.predict(X_test)

            probs = None

            if hasattr(model, "predict_proba"):

                pred_probs = model.predict_proba(X_test)

                if pred_probs.shape[1] == 2:
                    probs = pred_probs[:, 1]

                else:
                     probs = None

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

    # Leaderboard Cards
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
                <h3 style="color:#111827;">{medals[i]} {row['Model']}</h3>
                <h1 style="color:#2563eb;">{row['Accuracy']:.3f}</h1>
                <p style="color:#64748b;">Accuracy</p>
                <hr>
                <p style="color:#334155;">Precision: {row['Precision']:.3f}</p>
                <p style="color:#334155;">Recall: {row['Recall']:.3f}</p>
                <p style="color:#334155;">F1 Score: {row['F1 Score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Heatmap
    st.subheader("Metric Heatmap")

    fig_heat = px.imshow(
        results.set_index("Model"),
        text_auto=True,
        aspect="auto",
        title="Performance Matrix"
    )

    fig_heat.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Radar
    st.subheader("Multi-Metric Radar Analysis")

    fig_radar = go.Figure()

    for _, row in results.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                row["Accuracy"],
                row["Precision"],
                row["Recall"],
                row["F1 Score"]
            ],
            theta=["Accuracy", "Precision", "Recall", "F1"],
            fill='toself',
            name=row["Model"]
        ))

    fig_radar.update_layout(template="plotly_white", height=520)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Accuracy Bar
    st.subheader("Accuracy Benchmark")

    fig_bar = px.bar(
        results,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        text="Accuracy"
    )

    fig_bar.update_layout(template="plotly_white", height=420)
    fig_bar.update_traces(texttemplate="%{text:.3f}")

    st.plotly_chart(fig_bar, use_container_width=True)

    # Recommendation
    best = results.iloc[0]

    st.success(f"""
Recommended Production Model: **{best['Model']}**

Why this model was selected:
- Highest predictive accuracy
- Strong metric consistency
- Reliable churn classification performance
- Best deployment readiness
""")
