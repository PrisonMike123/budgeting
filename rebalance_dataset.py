import pandas as pd

# Load monthly dataset
df = pd.read_csv("data/monthly_financial_data.csv")

# Quick check
print(df.head())
print(df.columns)
