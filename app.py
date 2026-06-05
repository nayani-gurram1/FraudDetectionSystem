import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.saving import register_keras_serializable

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Fraud Detection Intelligence System")

# ==================================================
# ✅ REGISTER CUSTOM LAYER (CRITICAL FIX)
# ==================================================
@register_keras_serializable()
class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(10000, (2*(i//2))/d_model)
        angle_rads = pos * angle_rates

        PE = np.zeros((seq_len, d_model))
        PE[:, 0::2] = np.sin(angle_rads[:, 0::2])
        PE[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(PE, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding


# ==================================================
# ✅ LOAD MODEL (FIXED)
# ==================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fraud_model.h5",   # 🔥 USE .h5 FORMAT
        custom_objects={"PositionalEncodingLayer": PositionalEncodingLayer},
        compile=False
    )

model = load_model()


# ==================================================
# ✅ FILE UPLOAD
# ==================================================
file = st.file_uploader("📂 Upload creditcard.csv", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # ==================================================
    # ✅ VALIDATION
    # ==================================================
    if "Class" not in df.columns:
        st.error("❌ Dataset must contain 'Class' column")
        st.stop()

    # Reduce size (for Streamlit speed)
    df = df.head(5000)

    # Sort if Time exists
    if "Time" in df.columns:
        df = df.sort_values("Time")

    # ==================================================
    # 🔄 SEQUENCE CREATION
    # ==================================================
    seq_len = 5
    X_seq = []

    for i in range(len(df) - seq_len):
        X_seq.append(
            df.iloc[i:i+seq_len]
            .drop("Class", axis=1)
            .values
        )

    X_seq = np.array(X_seq)

    st.write("📦 Sequence Shape:", X_seq.shape)

    # ==================================================
    # ❌ SAFETY CHECK
    # ==================================================
    if len(X_seq) == 0:
        st.error("❌ Not enough data to create sequences")
        st.stop()

    # ==================================================
    # 🔮 PREDICTION
    # ==================================================
    with st.spinner("🔄 Running Model..."):
        preds = model.predict(X_seq)

    st.subheader("⚠️ Fraud Predictions")
    st.write(preds[:10])

    # ==================================================
    # 🔥 HIGH RISK DETECTION
    # ==================================================
    threshold = 0.5
    high_risk = np.where(preds > threshold)[0]

    st.subheader("🔥 High Risk Transactions")
    st.write(high_risk[:10])

    st.write(f"Total High Risk: {len(high_risk)}")

    # ==================================================
    # 📊 FRAUD DISTRIBUTION
    # ==================================================
    st.subheader("📊 Fraud Probability Distribution")

    fig1, ax1 = plt.subplots()
    ax1.hist(preds, bins=20)
    ax1.set_title("Fraud Score Distribution")

    st.pyplot(fig1)

    # ==================================================
    # 🧠 TRANSACTION IMPORTANCE
    # ==================================================
    st.subheader("🧠 Transaction Importance (Sequence)")

    sample = X_seq[0]
    importance = np.mean(sample, axis=1)

    fig2, ax2 = plt.subplots()
    ax2.bar(range(len(importance)), importance)
    ax2.set_title("Transaction Importance")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Score")

    st.pyplot(fig2)

else:
    st.info("📌 Please upload the dataset to begin.")
