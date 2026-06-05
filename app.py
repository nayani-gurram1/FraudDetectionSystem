import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="🚨",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🚨 Fraud Detection Intelligence System")
st.markdown("### AI-Based Transaction Risk Analysis (Demo Version)")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload creditcard.csv",
    type=["csv"]
)

# =====================================================
# MAIN APP
# =====================================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.write("Shape:", df.shape)

        # =====================================================
        # BASIC CHECK
        # =====================================================

        if "Class" not in df.columns:
            st.warning("⚠️ 'Class' column not found (Fraud labels missing)")

        # =====================================================
        # SORT DATA
        # =====================================================

        if "Time" in df.columns:
            df = df.sort_values("Time")

        # =====================================================
        # FEATURE SELECTION
        # =====================================================

        numeric_df = df.select_dtypes(include=[np.number])

        if "Class" in numeric_df.columns:
            features = numeric_df.drop("Class", axis=1)
        else:
            features = numeric_df

        # =====================================================
        # SEQUENCE CREATION
        # =====================================================

        seq_len = 5
        X = []

        for i in range(len(features) - seq_len):
            X.append(features.iloc[i:i+seq_len].values)

        X = np.array(X)

        st.write("Generated Sequences:", X.shape)

        if len(X) == 0:
            st.error("Not enough data to create sequences")
            st.stop()

        # =====================================================
        # 🚀 DEMO PREDICTIONS (NO MODEL)
        # =====================================================

        # Simulated fraud probability
        predictions = np.random.rand(len(X))

        # =====================================================
        # RESULTS
        # =====================================================

        results = df.iloc[seq_len:].copy()
        results["Fraud_Probability"] = predictions

        def classify(p):
            if p >= 0.8:
                return "High Risk"
            elif p >= 0.5:
                return "Medium Risk"
            else:
                return "Low Risk"

        results["Risk_Level"] = results["Fraud_Probability"].apply(classify)

        # =====================================================
        # METRICS
        # =====================================================

        st.subheader("📈 Fraud Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(results))

        col2.metric(
            "High Risk",
            len(results[results["Risk_Level"] == "High Risk"])
        )

        col3.metric(
            "Avg Fraud Score",
            round(results["Fraud_Probability"].mean(), 4)
        )

        # =====================================================
        # HIGH RISK TABLE
        # =====================================================

        st.subheader("🚨 High Risk Transactions")

        high_risk = results[results["Risk_Level"] == "High Risk"]

        st.dataframe(high_risk, use_container_width=True)

        # =====================================================
        # TREND GRAPH
        # =====================================================

        st.subheader("📊 Fraud Probability Trend")

        fig = px.line(results, y="Fraud_Probability")

        st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # PIE CHART
        # =====================================================

        st.subheader("📌 Risk Distribution")

        pie = results["Risk_Level"].value_counts().reset_index()
        pie.columns = ["Risk", "Count"]

        fig2 = px.pie(pie, names="Risk", values="Count")

        st.plotly_chart(fig2, use_container_width=True)

        # =====================================================
        # DOWNLOAD
        # =====================================================

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👆 Upload a CSV file to start analysis")
