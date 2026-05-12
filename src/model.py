import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier


def train_model(df):

    st.title("🧠 Advanced Model Training")

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

        results = []

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")
            f1 = f1_score(y_test, preds, average="weighted")

            results.append({
                "Model": name,
                "Accuracy": acc
            })

            # ---------------- MODEL CARD ----------------
            st.markdown("---")
            st.subheader(f"📌 {name}")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{acc:.2f}")
            c2.metric("Precision", f"{prec:.2f}")
            c3.metric("Recall", f"{rec:.2f}")
            c4.metric("F1 Score", f"{f1:.2f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)

            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=["No Churn", "Churn"],
                y=["No Churn", "Churn"]
            )

            st.plotly_chart(fig_cm, use_container_width=True)

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
                    orientation="h",
                    title=f"{name} Feature Importance"
                )

                st.plotly_chart(fig_feat, use_container_width=True)

            # Evaluation text
            if acc > 0.85:
                st.success(f"{name} performs excellently on customer churn prediction.")
            elif acc > 0.75:
                st.info(f"{name} gives good performance.")
            else:
                st.warning(f"{name} may need tuning.")

        st.session_state.results = pd.DataFrame(results)
