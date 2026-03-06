import streamlit as st
import pandas as pd

st.title("Insight to Innovation Engine")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.dataframe(df)

    st.success("Dataset uploaded successfully. Analysis ready.")