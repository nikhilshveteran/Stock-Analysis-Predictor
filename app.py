import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load your CSV
df = pd.read_csv("nifty50_closing_prices.csv")

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Create the Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Nifty 50 Stock Analysis Predictor", style={'textAlign': 'center'}),
    
    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns if col != 'Date'],
        value='RELIANCE.NS',  # Default selection
        clearable=False
    ),

    dcc.Graph(id='stock-trend-graph'),

    dcc.Graph(id='bar-chart')
])

# Callbacks to update graphs
@app.callback(
    Output('stock-trend-graph', 'figure'),
    Input('stock-dropdown', 'value')
)
def update_line_chart(selected_stock):
    fig = px.line(df, x="Date", y=selected_stock, title=f"{selected_stock} Stock Trend")
    return fig

@app.callback(
    Output('bar-chart', 'figure'),
    Input('stock-dropdown', 'value')
)
def update_bar_chart(selected_stock):
    fig = px.bar(df, x="Date", y=selected_stock, title=f"{selected_stock} Revenue & Risk Analysis")
    return fig

# Run server
if __name__ == '__main__':
    app.run_server(debug=True)
