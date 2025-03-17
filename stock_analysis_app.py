import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime

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

# Historical Trend Visualization
st.subheader("Historical Trends")
fig, ax = plt.subplots(figsize=(10, 5))
for stock in selected_stocks:
    ax.plot(df_selected['Date'], df_selected[stock], label=stock)
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.set_title("Stock Price Over Time")
ax.legend()
plt.xticks(rotation=45)  # Improve date readability
st.pyplot(fig)

# Bar Chart for better comparison
st.subheader("Stock Comparison - Bar Graph")
latest_prices = df_selected.iloc[-1, 1:]
fig, ax = plt.subplots()
sns.barplot(x=latest_prices.index, y=latest_prices.values, ax=ax)
ax.set_xlabel("Stock")
ax.set_ylabel("Latest Closing Price")
ax.set_title("Latest Stock Prices")
plt.xticks(rotation=45)
st.pyplot(fig)

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
