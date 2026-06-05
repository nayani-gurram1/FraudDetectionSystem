import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.title("💳 Fraud Detection Intelligence System")

# =========================
# ✅ Custom Layer (IMPORTANT FIX)
# =========================
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


# =========================
# ✅ Load Model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fraud_model.keras",
        custom_objects={
            "PositionalEncodingLayer": PositionalEncodingLayer
        },
        compile=False   # 🔥 IMPORTANT FIX
    )

model = load_model()

# =========================
# ✅ File Upload
# =========================
file = st.file_uploader("Upload creditcard.csv")

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Uploaded Data")
    st.write(df.head())

    # =========================
    # ⚠️ Reduce size for speed
    # =========================
    df = df.head(5000)

    df = df.sort_values("Time")

    # =========================
    # 🔄 Create Sequences
    # =========================
    seq_len = 5
    X_seq = []

    for i in range(len(df) - seq_len):
        X_seq.append(df.iloc[i:i+seq_len].drop("Class", axis=1).values)

    X_seq = np.array(X_seq)

    st.write("Total Sequences:", X_seq.shape[0])

    # =========================
    # 🔮 Prediction
    # =========================
    with st.spinner("Predicting..."):
        preds = model.predict(X_seq)

    st.subheader("⚠️ Fraud Predictions")
    st.write(preds[:10])

    # =========================
    # 🔥 High Risk Transactions
    # =========================
    threshold = 0.5
    high_risk = np.where(preds > threshold)[0]

    st.subheader("🔥 High Risk Transactions")
    st.write(high_risk[:10])

    # =========================
    # 🧠 Visualization
    # =========================
    st.subheader("🧠 Transaction Importance")

    sample = X_seq[0]
    importance = np.mean(sample, axis=1)

    fig, ax = plt.subplots()
    ax.bar(range(len(importance)), importance)
    ax.set_title("Transaction Importance")
    ax.set_xlabel("Transaction Index")
    ax.set_ylabel("Score")

    st.pyplot(fig)
