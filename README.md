# ML Lifecycle App
# Developed by Raazik

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ML App - Raazik")

st.title("ML Lifecycle App")
st.markdown("### Developed by Raazik")
st.markdown("---")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("1. Data Preview")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.subheader("2. Data Info")
    st.write(df.describe())

    st.subheader("3. Train Model")
    if st.button("Train Model"):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        st.success(f"Accuracy: {acc*100:.2f}%")
        st.balloons()
else:
    st.info("Please upload a CSV file")

st.markdown("---")
st.write("© 2026 | Developed by Raazik")
