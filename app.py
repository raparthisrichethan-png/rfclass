import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Random Forest Classifier",
    layout="wide"
)

# =========================================
# LOAD CSS
# =========================================

def load_css(file_name):

    with open(file_name) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# =========================================
# TITLE
# =========================================

st.markdown("""
<div class="card">
<h1>Heart Disease Prediction</h1>
<p>Random Forest Classification App</p>
</div>
""", unsafe_allow_html=True)

# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv("heart.csv")

# =========================================
# DATASET OVERVIEW
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# EDA
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Target Distribution")

fig1, ax1 = plt.subplots()

sns.countplot(
    x="target",
    data=df,
    ax=ax1
)

st.pyplot(fig1)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# CORRELATION
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=False,
    cmap="coolwarm",
    ax=ax2
)

st.pyplot(fig2)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# FEATURES & TARGET
# =========================================

X = df.drop("target", axis=1)

y = df["target"]

# =========================================
# SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# SCALING
# =========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =========================================
# MODEL TRAINING
# =========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================
# PREDICTIONS
# =========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# =========================================
# METRICS
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Model Accuracy")

st.metric(
    "Accuracy",
    f"{accuracy:.3f}"
)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# USER INPUTS
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Predict Heart Disease")

input_values = []

for col in X.columns:

    value = st.number_input(
        f"{col}",
        value=float(df[col].mean())
    )

    input_values.append(value)

# =========================================
# PREDICTION
# =========================================

if st.button("Predict"):

    input_df = pd.DataFrame(
        [input_values],
        columns=X.columns
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(
        input_scaled
    )

    if prediction[0] == 1:

        st.markdown(
            '''
            <div class="prediction-box">
            Heart Disease Detected
            </div>
            ''',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '''
            <div class="prediction-box">
            No Heart Disease
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.subheader("Prediction Probability")

    st.write(
        f"Class 0 Probability: {probability[0][0]:.4f}"
    )

    st.write(
        f"Class 1 Probability: {probability[0][1]:.4f}"
    )

st.markdown("</div>", unsafe_allow_html=True)