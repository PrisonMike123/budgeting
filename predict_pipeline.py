import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from visualizations import (
    create_financial_health_pie_chart,
    create_income_savings_histograms,
    create_correlation_heatmap,
    create_confusion_matrix_heatmap,
    create_actual_vs_predicted_plot,
    create_metrics_bar_chart,
    create_sample_predictions_table,
    create_methodology_visualizations
)

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
# Create output directories
plot_dir = os.path.join('static', 'images', 'plots')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(os.path.join(plot_dir, 'user_plots'), exist_ok=True)

# Convert to datetime for plotting
if 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
else:
    # Create synthetic months if not present
    df['month'] = pd.date_range(start="2024-01-01", periods=len(df), freq="M")

users = df['user_id'].unique()

# -----------------------------
# 9. Generate individual user trend plots
# -----------------------------
print("📊 Generating individual user plots...")
for user in users:
    user_data = df[df['user_id'] == user]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Income vs Expenses
    ax1.plot(user_data['month'], user_data['total_income'], 'g-', label='Income')
    ax1.plot(user_data['month'], user_data['total_expenses'], 'r-', label='Expenses')
    ax1.set_title(f'Monthly Income vs Expenses for {user}')
    ax1.set_ylabel('Amount ($)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: Savings Rate
    ax2.plot(user_data['month'], user_data['savings_ratio']*100, 'b-', label='Savings Rate')
    ax2.axhline(y=20, color='r', linestyle='--', label='Recommended (20%)')
    ax2.set_title(f'Monthly Savings Rate for {user}')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Savings Rate (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{user}_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("✅ Individual user plots saved.\n")

# -----------------------------
# 11. Generate Visualizations
# -----------------------------
print("📊 Generating visualizations...")

# 0. Methodology Visualizations
print("\n📝 Creating methodology visualizations...")
create_methodology_visualizations(save_path=plot_dir)

# 1. Financial Health Pie Chart
if 'financial_health' in df.columns:
    create_financial_health_pie_chart(df, save_path=plot_dir)

# 2. Income and Savings Histograms
if all(col in df.columns for col in ['total_income', 'savings_ratio']):
    create_income_savings_histograms(df, save_path=plot_dir)

# 3. Correlation Heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    create_correlation_heatmap(df[numeric_cols], save_path=plot_dir)

# 4. Confusion Matrix and Classification Report
if 'financial_health' in df.columns and 'predicted_financial_health' in df.columns:
    # Generate classification report and convert to dict
    report = classification_report(
        df['financial_health'],
        df['predicted_financial_health'],
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    # Print classification report
    print("\n=== Classification Report ===")
    print(classification_report(
        df['financial_health'],
        df['predicted_financial_health'],
        target_names=label_encoder.classes_
    ))
    
    # Create metrics bar chart
    create_metrics_bar_chart(report, save_path=plot_dir)
    
    # Create sample predictions table
    create_sample_predictions_table(df, n_samples=10, save_path=plot_dir)
    
    # Confusion Matrix
    create_confusion_matrix_heatmap(
        y_true=df['financial_health'],
        y_pred=df['predicted_financial_health'],
        classes=label_encoder.classes_,
        save_path=plot_dir
    )

# 5. Actual vs Predicted Values (for regression)
if 'total_expenses' in df.columns and 'predicted_expense' in df.columns:
    print("\n=== Regression Metrics ===")
    mse = mean_squared_error(df['total_expenses'], df['predicted_expense'])
    r2 = r2_score(df['total_expenses'], df['predicted_expense'])
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    create_actual_vs_predicted_plot(
        y_true=df['total_expenses'],
        y_pred=df['predicted_expense'],
        save_path=plot_dir
    )

# 6. Combined Savings Overview
print("\n✅ All visualizations saved to:")
print(f"   - {os.path.abspath(plot_dir)}/")

# Create combined savings overview
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
plt.savefig(os.path.join(plot_dir, "combined_savings_overview.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Combined dashboard saved in '{plot_dir}' folder.\n")
print("🎉 All steps complete. Predictions and plots ready.")
