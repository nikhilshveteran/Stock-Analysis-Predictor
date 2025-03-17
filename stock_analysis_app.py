import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
import datetime

# Streamlit Page Configuration
st.set_page_config(page_title="Stock Analysis Predictor", layout="wide")

# Upload CSV File
st.sidebar.title("Upload Nifty 50 CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File Uploaded Successfully!")
    
    # Display File Name and Path
    st.sidebar.write(f"File Name: {uploaded_file.name}")

    # Select a Stock from Dropdown
    stock_list = list(df.columns[1:])  # Exclude 'Date' column
    selected_stock = st.sidebar.selectbox("Select a Stock:", stock_list)

    # Convert 'Date' Column to Datetime Format
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Handle Missing Values
    df.fillna(method='ffill', inplace=True)

    # Historical Stock Trend
    st.title(f"Stock Analysis for {selected_stock}")
    st.subheader("Historical Trends")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[selected_stock], label=f'{selected_stock} Price', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

    # Bar Chart of Monthly Returns
    st.subheader("Monthly Average Price")
    df_monthly = df[selected_stock].resample('M').mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    df_monthly.plot(kind='bar', ax=ax, color='orange')
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Price")
    st.pyplot(fig)

    # Prediction Model
    st.subheader("Stock Price Prediction for Next 30 Days")
    
    df['Days'] = np.arange(len(df))  # Convert Date to Numeric

    # Train Linear Regression Model
    X = df[['Days']]
    y = df[selected_stock]
    model = LinearRegression()
    model.fit(X, y)

    # Predict Future Prices
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    future_prices = model.predict(future_days)

    # Plot Predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[selected_stock], label="Actual Prices", color='blue')
    future_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(1, 31)]
    ax.plot(future_dates, future_prices, label="Predicted Prices", linestyle="dashed", color='red')
    ax.legend()
    st.pyplot(fig)

    # Display Summary
    st.subheader("Prediction Summary")
    st.write(f"Expected Price After 30 Days: ₹{future_prices[-1]:.2f}")

else:
    st.warning("Please upload a CSV file to proceed.")
