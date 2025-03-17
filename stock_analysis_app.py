import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Load the dataset
CSV_FILE = "nifty50_closing_prices.csv"

def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        st.error(f"CSV file not found: {CSV_FILE}. Please check the file location.")
        return None

data = load_data()

if data is not None:
    st.title("Stock Analysis Predictor")
    st.sidebar.header("Select Stock")
    stock_list = data.columns[1:]
    selected_stock = st.sidebar.selectbox("Choose a stock:", stock_list)

    # Display historical trends
    st.subheader(f"Historical Trend of {selected_stock}")
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data[selected_stock], label=selected_stock, color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.legend()
    st.pyplot(fig)

    # Future Prediction using Linear Regression
    st.subheader("Future Prediction")
    try:
        stock_prices = data[selected_stock].dropna().values.reshape(-1, 1)
        days = np.arange(len(stock_prices)).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(days, stock_prices)
        
        future_days = np.arange(len(stock_prices), len(stock_prices) + 30).reshape(-1, 1)
        predicted_prices = model.predict(future_days)
        
        fig_pred, ax_pred = plt.subplots()
        ax_pred.plot(days, stock_prices, label="Historical Prices", color='blue')
        ax_pred.plot(future_days, predicted_prices, label="Predicted Prices", color='red', linestyle='dashed')
        ax_pred.set_xlabel("Days")
        ax_pred.set_ylabel("Stock Price")
        ax_pred.legend()
        st.pyplot(fig_pred)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
