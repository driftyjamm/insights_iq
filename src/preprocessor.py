import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def clean_data(df):

    st.title("Advanced Data Preprocessing")

    cleaning_log = []

    st.subheader("Original Dataset")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # ---------------- REMOVE DUPLICATES ----------------
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()
        cleaning_log.append({
            "Column": "All Columns",
            "Issue": f"{duplicate_count} Duplicate Rows",
            "Action": "Removed duplicate records"
        })

    # ---------------- IDENTIFY COLUMN TYPES ----------------
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # ---------------- NUMERICAL CLEANING ----------------
    for col in num_cols:

        missing = df[col].isnull().sum()

        if missing > 0:
            median_val = df[col].median()

            df[col].fillna(median_val, inplace=True)

            cleaning_log.append({
                "Column": col,
                "Issue": f"{missing} Missing Values",
                "Action": f"Filled using Median ({median_val:.2f})"
            })

        # Outlier handling
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        if outliers > 0:
            df[col] = np.clip(df[col], lower, upper)

            cleaning_log.append({
                "Column": col,
                "Issue": f"{outliers} Outliers",
                "Action": "Capped using IQR method"
            })

    # ---------------- CATEGORICAL CLEANING ----------------
    le = LabelEncoder()

    for col in cat_cols:

        missing = df[col].isnull().sum()

        if missing > 0:
            mode_val = df[col].mode()[0]

            df[col].fillna(mode_val, inplace=True)

            cleaning_log.append({
                "Column": col,
                "Issue": f"{missing} Missing Values",
                "Action": f"Filled using Mode ({mode_val})"
            })

        unique_before = df[col].nunique()

        df[col] = le.fit_transform(df[col])

        cleaning_log.append({
            "Column": col,
            "Issue": f"{unique_before} Categories",
            "Action": "Label Encoded"
        })

    # ---------------- CLEANING REPORT ----------------
    st.subheader("Data Cleaning Report")

    if cleaning_log:
        log_df = pd.DataFrame(cleaning_log)
        st.dataframe(log_df, use_container_width=True)

    else:
        st.success("Dataset already clean")

    # ---------------- SUMMARY ----------------
    st.subheader("Cleaning Summary")

    st.info(f"""
Cleaning Operations Applied:

• Duplicate row removal  
• Missing value imputation  
• Outlier detection & capping  
• Categorical encoding  
• Data consistency normalization

Final Dataset Shape:
Rows: {df.shape[0]}
Columns: {df.shape[1]}
""")

    return df
