# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from custom_backend_model import load_and_predict_from_csv

st.set_page_config(layout="wide")
st.title("📈 AI Stock Market Predictor (Custom Backend)")

# User Inputs
symbol = st.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA, GLD)", "GLD")
timeframe = st.selectbox("Select Timeframe", ['1h', '4h', '1d', '1w'])

# Notify user: model expects preprocessed CSVs (e.g., data_1h.csv)
st.markdown("""
ℹ️ This tool assumes your data has already been fetched and processed by the backend pipeline.<br>
Make sure you have files like `data_1h.csv`, `data_1d.csv`, etc. ready before running predictions.
""", unsafe_allow_html=True)

if st.button("Run Prediction"):
    with st.spinner(f"Loading predictions for {symbol} at {timeframe} timeframe..."):
        try:
            fig, predictions_df, metrics = load_and_predict_from_csv(timeframe)
            st.pyplot(fig)
            avg_price = predictions_df['actual_close'].mean()
            price_accuracy = 100 - (metrics['mae'] / avg_price * 100)
            direction_accuracy = metrics['dir_acc']

            st.write("Metrics")
            st.write(f"Price Accuracy: {price_accuracy:.2f}%")
            st.write(f"Direction Accuracy: {direction_accuracy:.2f}%")
            
        except FileNotFoundError:
            st.error(f"❌ File not found: `data_{timeframe}.csv`. Please run the data pipeline first.")
            
