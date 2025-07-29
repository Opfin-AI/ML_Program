# streamlit_app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from my_model_module import run_model_for_streamlit

st.set_page_config(layout="wide")
st.title("📈 AI Stock Market Predictor")

# User inputs
symbol = st.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA, MSFT)", "AAPL")
timeframe = st.selectbox("Select Timeframe", ['1h', '4h', '1d', '1w'])
days = st.slider("Number of days of data to use", min_value=30,
max_value=730, value=180)

if st.button("Fetch & Predict"):
   with st.spinner("Downloading and processing data..."):
       interval_map = {'1h': '60m', '4h': '240m', '1d': '1d', '1w': '1wk'}
       interval = interval_map[timeframe]

       # Download data
       df = yf.download(symbol, period=f"{days}d", interval=interval)

       # Reset and rename index to standard 'timestamp'
       df.reset_index(inplace=True)
       df.rename(columns={"Date": "timestamp", "Datetime":
"timestamp"}, inplace=True)

       # Safety check for timestamp column
       if 'timestamp' not in df.columns:
           st.error("❌ Timestamp column not found in the downloaded data. Check the interval or ticker symbol.")
           st.stop()

   # Show error if no data
   if df.empty:
       st.error("No data found. Try a different symbol or time range.")
   else:
       st.subheader("Preview of Downloaded Data")
       st.dataframe(df.tail())

       with st.spinner("Running model..."):
           try:
               fig, predictions_df, metrics = run_model_for_streamlit(df, timeframe)
           except Exception as e:
               st.error(f"❌ Model error: {e}")
               st.stop()

       st.subheader("Prediction Plot")
       st.pyplot(fig)

       st.subheader("Latest Next-Bar Predictions")
       st.dataframe(predictions_df.tail())

       st.subheader("Performance Metrics")
       st.markdown(f"**MAE:** {metrics['mae']:.3f}%  ")
       st.markdown(f"**RMSE:** {metrics['rmse']:.3f}%  ")
       st.markdown(f"**Directional Accuracy:** {metrics['dir_acc']:.2f}%")

    
