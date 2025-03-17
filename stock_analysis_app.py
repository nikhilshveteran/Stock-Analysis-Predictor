import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import plotly.express as px
import numpy as np

# Load dataset
csv_file = "nifty50_closing_prices.csv"
df = pd.read_csv(csv_file)
df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date column is in datetime format

# Streamlit UI setup
st.title("Stock Analysis Predictor")
st.sidebar.header("Stock Selection")

# Allow multiple stock selection
selected_stocks = st.sidebar.multiselect("Select Stocks", df.columns[1:], default=df.columns[1])

# Filter data based on selected stocks
df_selected = df[['Date'] + selected_stocks]

# Historical Trend Visualization with hover feature
st.subheader("Historical Trends")
fig = px.line(df_selected, x='Date', y=selected_stocks, title="Stock Price Over Time", labels={'value': 'Stock Price', 'Date': 'Date'})
fig.update_traces(mode='lines+markers', hovertemplate='%{x}<br>%{y}')
st.plotly_chart(fig)

# Bar Chart for better comparison
st.subheader("Stock Comparison - Bar Graph")
latest_prices = df_selected.iloc[-1, 1:]
fig = px.bar(x=latest_prices.index, y=latest_prices.values, labels={'x': 'Stock', 'y': 'Latest Closing Price'}, title="Latest Stock Prices")
st.plotly_chart(fig)

# Risk and Volatility Calculation
def calculate_risk_volatility(stock):
    returns = df[stock].pct_change().dropna()
    risk = np.std(returns)
    volatility = risk * np.sqrt(252)  # Annualized volatility
    return risk, volatility

st.subheader("Risk & Volatility Analysis")
risk_volatility_results = {}
for stock in selected_stocks:
    risk, volatility = calculate_risk_volatility(stock)
    risk_volatility_results[stock] = {'Risk': risk, 'Volatility': volatility}
st.write("Risk & Volatility Data:", risk_volatility_results)

# Future Prediction using Yahoo Finance
def predict_future(stock):
    ticker = yf.Ticker(stock)
    hist = ticker.history(period="1y")
    last_price = hist['Close'].iloc[-1]
    predicted_price = last_price * 1.05  # Example: Predicting 5% growth
    return predicted_price

st.subheader("Future Price Prediction")
prediction_results = {}
for stock in selected_stocks:
    predicted_price = predict_future(stock)
    prediction_results[stock] = predicted_price
st.write("Predicted Prices for Next Period:", prediction_results)
