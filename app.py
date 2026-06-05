import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import os

# ============================================
# 🎯 PAGE SETUP
# ============================================
st.set_page_config(
    page_title="AI Fraud Monitor",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Fraud Detection System")
st.caption("Detect suspicious transactions using deep learning")

# ============================================
# 📌 SIDEBAR INFO
# ============================================
st.sidebar.header("System Status")

st.sidebar.write("📁 Working Directory:")
st.sidebar.code(os.getcwd())

st.sidebar.write("📦 Files Available:")
st.sidebar.write(os.listdir())

# ============================================
# 🧠 MODEL LOADING
# ============================================
MODEL_FILE = "attention_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.error("❌ Model file not found")
        st.stop()
    return tf.keras.models.load_model(MODEL_FILE, compile=False)

try:
    model = load_model()
    st.sidebar.success("✅ Model Ready")
except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()

# ============================================
# 📂 FILE INPUT
# ============================================
st.subheader("📂 Upload Transaction Dataset")

file = st.file_uploader("Upload CSV", type=["csv"])

# ============================================
# 🚀 MAIN PIPELINE
# ============================================
if file:

    df = pd.read_csv(file)

    st.subheader("📊 Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("Shape:", df.shape)

    # ============================================
    # 🔍 FEATURE SELECTION
    # ============================================
    num_df = df.select_dtypes(include=np.number)

    if num_df.shape[1] == 0:
        st.error("No numeric data found")
        st.stop()

    st.subheader("🔢 Features Used")
    st.write(list(num_df.columns))

    # ============================================
    # 📏 MODEL REQUIREMENTS
    # ============================================
    input_shape = model.input_shape
    seq_len = input_shape[1]
    feat_count = input_shape[2]

    st.info(f"Model expects {feat_count} features & sequence length {seq_len}")

    if num_df.shape[1] != feat_count:
        st.error("Feature mismatch between model & dataset")
        st.stop()

    # ============================================
    # 🔄 SEQUENCE BUILDING
    # ============================================
    data = num_df.values
    sequences = []

    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])

    X = np.array(sequences)

    st.write("Generated Sequences:", X.shape)

    if len(X) == 0:
        st.error("Not enough data for sequence creation")
        st.stop()

    # ============================================
    # 🔮 MODEL PREDICTION
    # ============================================
    with st.spinner("Analyzing transactions..."):
        preds = model.predict(X, verbose=0).flatten()

    # ============================================
    # 📊 RESULTS CREATION
    # ============================================
    result_df = df.iloc[seq_len:].copy()
    result_df["Fraud_Score"] = preds

    # Risk labeling
    def risk_label(x):
        if x > 0.8:
            return "🔴 High"
        elif x > 0.5:
            return "🟡 Medium"
        else:
            return "🟢 Low"

    result_df["Risk"] = result_df["Fraud_Score"].apply(risk_label)

    # ============================================
    # 📌 METRICS
    # ============================================
    st.subheader("📈 Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", len(result_df))
    c2.metric("High Risk", (result_df["Risk"] == "🔴 High").sum())
    c3.metric("Avg Score", round(result_df["Fraud_Score"].mean(), 4))

    # ============================================
    # 🚨 HIGH RISK DATA
    # ============================================
    st.subheader("🚨 Suspicious Transactions")

    high_risk = result_df[result_df["Risk"] == "🔴 High"]

    st.dataframe(high_risk, use_container_width=True)

    # ============================================
    # 📉 TREND GRAPH
    # ============================================
    st.subheader("📉 Fraud Score Trend")

    fig1 = px.line(result_df, y="Fraud_Score")
    st.plotly_chart(fig1, use_container_width=True)

    # ============================================
    # 🥧 DISTRIBUTION
    # ============================================
    st.subheader("🥧 Risk Breakdown")

    dist = result_df["Risk"].value_counts().reset_index()
    dist.columns = ["Risk", "Count"]

    fig2 = px.pie(dist, names="Risk", values="Count")
    st.plotly_chart(fig2, use_container_width=True)

    # ============================================
    # 📥 DOWNLOAD
    # ============================================
    st.subheader("⬇️ Export Results")

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "fraud_results.csv",
        "text/csv"
    )

else:
    st.info("📌 Upload a CSV file to begin analysis")