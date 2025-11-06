import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# -----------------------------
# 1. Load models
# -----------------------------
print("🔍 Loading models...")
reg_model = joblib.load("models/linear_regression.pkl")
clf_model = joblib.load("models/financial_health_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
print("✅ Models loaded successfully.\n")

# -----------------------------
# 2. Load dataset
# -----------------------------
print("📂 Loading data...")
df = pd.read_csv("data/financial_features.csv")
print("✅ Data loaded. Sample:")
print(df.head(), "\n")

# -----------------------------
# 3. Prepare data for classification
# -----------------------------
print("🧮 Preparing data for classification...")
# These are the features the scaler was trained on
classification_features = [
    'total_income', 'total_expenses', 'savings', 
    'savings_ratio', 'debt_ratio', 'expense_ratio'
]

# Prepare data for classification
X_clf = df[classification_features].copy()
X_clf_scaled = scaler.transform(X_clf)

# Prepare data for regression (using different features)
regression_features = [
    'age', 'family_size', 'total_income', 'debt_ratio',
    'expense_ratio', 'savings_ratio', 'credit_score'
]

# Convert age from string to numeric if needed
def convert_age(age):
    if isinstance(age, str) and '-' in age:
        start, end = map(int, age.split('-'))
        return (start + end) / 2
    try:
        return float(age)
    except (ValueError, TypeError):
        return None

X_reg = df[regression_features].copy()
if 'age' in X_reg.columns:
    X_reg['age'] = X_reg['age'].apply(convert_age)

# Fill any remaining NaNs with column means
X_reg = X_reg.fillna(X_reg.mean())

print("\n=== Sample of prepared data ===")
print("Regression features (first 5 rows):")
print(X_reg.head())
print("\nClassification features (first 5 rows):")
print(X_clf.head())

# -----------------------------
# 4. Predict expenses
# -----------------------------
print("💸 Making regression predictions...")
df['predicted_expense'] = reg_model.predict(X_reg)

# -----------------------------
# 5. Predict financial health
# -----------------------------
print("🏦 Making classification predictions...")
# We already have X_clf_scaled from earlier
# Make predictions using the classifier
health_predictions = clf_model.predict(X_clf_scaled)
# Convert numeric predictions back to original labels
df['predicted_financial_health'] = label_encoder.inverse_transform(health_predictions)

# -----------------------------
# 7. Save predictions
# -----------------------------
output_file = "data/final_predictions.csv"
df.to_csv(output_file, index=False)
print(f"✅ Predictions saved to {output_file}\n")

# -----------------------------
# 8. Visualization setup
# -----------------------------
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Convert to datetime for plotting
if 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
else:
    # Create synthetic months if not present
    df['month'] = pd.date_range(start="2024-01-01", periods=len(df), freq="M")

users = df['user_id'].unique()

# -----------------------------
# 9. Plot individual user trends
# -----------------------------
print("📊 Generating individual user plots...")
for user in users:
    user_df = df[df['user_id'] == user].sort_values('month')
    
    plt.figure(figsize=(10, 5))
    plt.plot(user_df['month'], user_df['savings'], marker='o', linestyle='-', label='Actual Savings')
    plt.plot(user_df['month'], user_df['predicted_expense'], marker='x', linestyle='--', label='Predicted Expense')
    plt.title(f"User {user}: Savings vs Predicted Expense")
    plt.xlabel("Month")
    plt.ylabel("Amount")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{user}_trend.png")
    plt.close()

print("✅ Individual user plots saved.\n")

# -----------------------------
# 10. Combined Dashboard Overview
# -----------------------------
print("📊 Generating combined dashboard overview...")

plt.figure(figsize=(12, 6))
for user in users:
    user_df = df[df['user_id'] == user].sort_values('month')
    plt.plot(user_df['month'], user_df['savings'], marker='o', linestyle='-', alpha=0.3, color='blue')
    plt.plot(user_df['month'], user_df['predicted_expense'], marker='x', linestyle='--', alpha=0.3, color='orange')

plt.title("All Users: Actual Savings vs Predicted Expense")
plt.xlabel("Month")
plt.ylabel("Amount")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.legend(["Actual Savings", "Predicted Expense"], loc="upper left")
plt.tight_layout()
plt.savefig(f"{plot_dir}/combined_savings_overview.png")
plt.close()

# Distribution of predicted financial health
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='predicted_financial_health', palette=['green', 'orange', 'red'])
plt.title("Distribution of Predicted Financial Health Across Users")
plt.xlabel("Financial Health Category")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(f"{plot_dir}/financial_health_distribution.png")
plt.close()

print(f"✅ Combined dashboard saved in '{plot_dir}' folder.\n")

print("🎉 All steps complete. Predictions and plots ready.")
