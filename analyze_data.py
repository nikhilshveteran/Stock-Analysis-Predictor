import pandas as pd

# Define file path
file_path = r"C:\Users\Dell\OneDrive\Documents\Stock_Analysis_Predictor\nifty50_closing_prices.csv"

# Load CSV
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print("Error: CSV file not found. Check the file path.")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Print column names
print("Columns in CSV:", df.columns)
