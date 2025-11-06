import pandas as pd
import numpy as np

print("📂 Loading base dataset...")
df = pd.read_csv("data/final_predictions.csv")

months = pd.date_range(start="2024-01-01", periods=12, freq="M")
synthetic_rows = []

print("🧮 Generating monthly financial data...")

for _, row in df.iterrows():
    for month in months:
        income = row['total_income'] * np.random.uniform(0.9, 1.1)
        expenses = row['predicted_expense'] * np.random.uniform(0.85, 1.15)
        savings = income - expenses
        savings_ratio = max(savings / income, 0)
        debt_ratio = row['debt_ratio'] * np.random.uniform(0.9, 1.1)
        expense_ratio = expenses / income if income > 0 else 0

        synthetic_rows.append({
            'user_id': row['user_id'],
            'month': month.strftime('%Y-%m'),
            'total_income': round(income, 2),
            'total_expenses': round(expenses, 2),
            'savings': round(savings, 2),
            'savings_ratio': round(savings_ratio, 3),
            'debt_ratio': round(debt_ratio, 3),
            'expense_ratio': round(expense_ratio, 3),
            'predicted_financial_health': row['predicted_financial_health']
        })

monthly_df = pd.DataFrame(synthetic_rows)
monthly_df.to_csv("data/monthly_financial_data.csv", index=False)

print("✅ Synthetic monthly dataset created and saved as data/monthly_financial_data.csv")
print("Preview:")
print(monthly_df.head())
