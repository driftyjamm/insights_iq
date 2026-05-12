import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

import plotly.express as px
import plotly.figure_factory as ff

def clean_data(df):

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove empty columns
    df = df.dropna(axis=1, how="all")

    # Remove ID columns
    remove_cols = []

    for col in df.columns:
        if "id" in col.lower():
            remove_cols.append(col)

    df = df.drop(columns=remove_cols, errors="ignore")

    return df

def preprocess_data(df, target):

    X = df.drop(target, axis=1)
    y = df[target]

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    X = pd.get_dummies(X)

    return X, y

def train_model(df):

    st.title("🧠 Advanced Model Training")

    df = clean_data(df)

    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    X, y = preprocess_data(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if st.button("🚀 Train Advanced Models"):

        with st.spinner("Training models..."):

            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    random_state=42
                ),

                "Extra Trees": ExtraTreesClassifier(
                    n_estimators=200,
                    random_state=42
                ),

                "AdaBoost": AdaBoostClassifier(
                    n_estimators=150,
                    random_state=42
                )
            }

            results = []

            best_model = None
            best_score = 0

            for name, model in models.items():

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, average="weighted")
                rec = recall_score(y_test, preds, average="weighted")
                f1 = f1_score(y_test, preds, average="weighted")

                try:
                    auc = roc_auc_score(y_test, preds)
                except:
                    auc = 0

                results.append({
                    "Model": name,
                    "Accuracy": round(acc, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1": round(f1, 4),
                    "ROC-AUC": round(auc, 4)
                })

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_preds = preds
                    best_name = name

            result_df = pd.DataFrame(results)

            st.session_state.model = best_model
            st.session_state.columns = X.columns
            st.session_state.target = target
            st.session_state.df = df

        st.success("Models trained successfully")

        col1, col2, col3 = st.columns(3)

        col1.metric("Best Model", best_name)
        col2.metric("Accuracy", f"{best_score:.2f}")
        col3.metric("Models", "4")

        st.markdown("---")

        st.subheader("📋 Model Comparison")
        st.dataframe(result_df, use_container_width=True)

        fig = px.bar(
            result_df,
            x="Model",
            y="Accuracy",
            color="Accuracy",
            text="Accuracy",
            title="Accuracy Comparison"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(
            result_df,
            x="Model",
            y=["Precision", "Recall", "F1"],
            markers=True,
            title="Performance Metrics"
        )

        fig2.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig2, use_container_width=True)

        cm = confusion_matrix(y_test, best_preds)

        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["No Churn", "Churn"],
            y=["No Churn", "Churn"],
            colorscale="Blues"
        )

        fig_cm.update_layout(
            template="plotly_dark",
            title="Confusion Matrix",
            height=450
        )

        st.plotly_chart(fig_cm, use_container_width=True)

        if hasattr(best_model, "feature_importances_"):

            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": best_model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            ).head(10)

            fig3 = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Important Features"
            )

            fig3.update_layout(
                template="plotly_dark",
                height=500
            )

            st.plotly_chart(fig3, use_container_width=True)