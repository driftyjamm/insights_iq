import pandas as pd
import numpy as np
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_data(df):
    st.subheader("Advanced Data Preprocessing Pipeline")

    cleaning_log = []

    # ---------------- INITIAL INFO ----------------
    st.write("### Initial Dataset")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    original_shape = df.shape

    # ---------------- REMOVE DUPLICATES ----------------
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()
        cleaning_log.append(f"Removed {duplicate_count} duplicate rows")

    # ---------------- REMOVE EMPTY COLUMNS ----------------
    empty_cols = df.columns[df.isnull().all()].tolist()

    if empty_cols:
        df = df.drop(columns=empty_cols)
        cleaning_log.append(f"Removed empty columns: {', '.join(empty_cols)}")

    # ---------------- DATA TYPE FIXING ----------------
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
                cleaning_log.append(f"Converted {col} to numeric")
            except:
                pass

    # ---------------- IDENTIFY COLUMNS ----------------
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # ---------------- HANDLE MISSING VALUES ----------------
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        cleaning_log.append("Filled missing numeric values using median")

    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        cleaning_log.append("Filled missing categorical values using mode")

    # ---------------- OUTLIER HANDLING ----------------
    outlier_fixed = 0

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        before = ((df[col] < lower) | (df[col] > upper)).sum()

        df[col] = np.clip(df[col], lower, upper)

        outlier_fixed += before

    if outlier_fixed > 0:
        cleaning_log.append(f"Handled {outlier_fixed} outliers using IQR capping")

    # ---------------- ENCODE CATEGORICAL ----------------
    encoded_cols = []

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoded_cols.append(col)

    if encoded_cols:
        cleaning_log.append(f"Encoded categorical columns")

    # ---------------- FEATURE SCALING ----------------
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        cleaning_log.append("Applied Standard Scaling")

    # ---------------- CONSTANT COLUMN REMOVAL ----------------
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

    if constant_cols:
        df = df.drop(columns=constant_cols)
        cleaning_log.append(f"Removed constant columns")

    # ---------------- FINAL SUMMARY ----------------
    st.success("Preprocessing Completed Successfully")

    st.write("### Cleaning Actions Performed")

    for i, action in enumerate(cleaning_log, 1):
        st.write(f"{i}. {action}")

    st.write("### Final Dataset")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    # ---------------- BEFORE VS AFTER ----------------
    comparison = pd.DataFrame({
        "Stage": ["Before Cleaning", "After Cleaning"],
        "Rows": [original_shape[0], df.shape[0]],
        "Columns": [original_shape[1], df.shape[1]]
    })

    st.dataframe(comparison, use_container_width=True)

    # ---------------- DATA PREVIEW ----------------
    st.write("### Cleaned Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    return df
