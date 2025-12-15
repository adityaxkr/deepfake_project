import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- Config ---
st.set_page_config(page_title="📊 Feedback Review Dashboard", layout="wide")

st.title("📊 Feedback Review Dashboard for DeepFakeNet")
st.markdown("Analyze user feedback to improve model performance.")

# --- Load Feedback ---
feedback_file = "model_training/feedback_log.csv"

if not os.path.exists(feedback_file):
    st.warning("⚠️ No feedback data found.")
    st.stop()

df = pd.read_csv(feedback_file)

# --- Date Conversion ---
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- Filters ---
st.sidebar.header("🔎 Filters")

date_range = st.sidebar.date_input("📅 Select date range", [df["timestamp"].min(), df["timestamp"].max()])
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df_filtered = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

predicted = st.sidebar.multiselect("🔮 Predicted Label", options=df["predicted_label"].unique(), default=df["predicted_label"].unique())
correctness = st.sidebar.multiselect("✅ Was Prediction Correct?", options=df["was_correct"].unique(), default=df["was_correct"].unique())

df_filtered = df_filtered[
    (df_filtered["predicted_label"].isin(predicted)) &
    (df_filtered["was_correct"].isin(correctness))
]

# --- Summary Stats ---
st.subheader("📈 Summary Statistics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Feedback", len(df_filtered))
with col2:
    acc = (df_filtered["was_correct"] == "Yes").mean()
    st.metric("Model Accuracy (User-reported)", f"{acc*100:.2f}%")
with col3:
    wrong_preds = df_filtered[df_filtered["was_correct"] == "No"]
    st.metric("Incorrect Predictions", len(wrong_preds))

# --- Visualizations ---
st.subheader("📊 Visualizations")

col4, col5 = st.columns(2)
with col4:
    st.markdown("#### Prediction Distribution")
    st.bar_chart(df_filtered["predicted_label"].value_counts())

with col5:
    st.markdown("#### User Correctness")
    st.bar_chart(df_filtered["was_correct"].value_counts())

# --- Data Table ---
st.subheader("🧾 Feedback Table")

st.dataframe(df_filtered.sort_values(by="timestamp", ascending=False), use_container_width=True)

# --- Download CSV ---
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Filtered Feedback CSV", csv, "filtered_feedback.csv", "text/csv")
