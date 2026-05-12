import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def clean_data(df):

    st.write("### Initial Dataset Shape")
    st.write(df.shape)

    # Remove duplicates
    df = df.drop_duplicates()

    # Missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Encode categorical
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    st.success("Cleaning Completed")

    return df