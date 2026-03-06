import streamlit as st
import pandas as pd
import subprocess
import sys
import os

st.title("Insight to Innovation Engine")

uploaded_file = st.file_uploader("Upload Review Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    # limit dataset to avoid memory crash
    if len(df) > 200:
        df = df.sample(200, random_state=42)

    import app

    opportunities, meta = app.build_opportunities(df)

    st.subheader("Top Opportunities")

    for o in opportunities:
        st.write(o)