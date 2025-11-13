import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
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

def convert_age(age):
    """Convert age to appropriate bins."""
    if pd.isna(age):
        return 3  # Default to middle age group if missing
    if age < 30:
        return 1  # Young
    elif age < 50:
        return 2  # Middle-aged
    else:
        return 3  # Senior

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run financial health prediction pipeline')
parser.add_argument('--skip-user-plots', action='store_true', 
                    help='Skip generating individual user plots')
args = parser.parse_args()

# -----------------------------
# 1. Load models
# -----------------------------
# Load models
models_dir = "models/"
reg_model_path = os.path.join(models_dir, "xgboost/xgboost_regressor.pkl")
clf_model_path = os.path.join(models_dir, "xgboost/xgboost_classifier.pkl")

# Check if all required files exist
for path in [reg_model_path, clf_model_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# Load models
print("🔍 Loading XGBoost models...")
reg_model = joblib.load(reg_model_path)
clf_model = joblib.load(clf_model_path)

# Load label encoder for classification
label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
print("✅ XGBoost models loaded successfully.\n")

# -----------------------------
# 2. Load dataset
# -----------------------------
print("📂 Loading data...")
df = pd.read_csv("data/financial_features_large.csv")
print("✅ Data loaded. Sample:")
print(f"Total users: {len(df['user_id'].unique()):,}")
print(f"Data shape: {df.shape}")
print("\nSample data:")
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

# Prepare data for classification (no scaling needed for XGBoost)
X_clf = df[classification_features].copy()

# Prepare regression features (ensure same order as training)
regression_features = ['age', 'family_size', 'total_income', 'debt_ratio', 'expense_ratio', 'savings_ratio', 'credit_score']

# Make sure all required features exist
missing_features = [f for f in regression_features if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing required features: {missing_features}")

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
# 4. Make Predictions
# -----------------------------
print("💸 Making regression predictions...")
try:
    # Make sure we're using the same feature order as during training
    X_reg_ordered = X_reg[regression_features]
    X_reg_df = pd.DataFrame(X_reg_ordered, columns=regression_features)
    
    # Make prediction (XGBoost doesn't need scaling)
    predicted_expense = reg_model.predict(X_reg_df)
    predicted_expense = [max(0, x) for x in predicted_expense]  # Ensure non-negative
    df['predicted_expense'] = predicted_expense
    if 'total_expenses' in df.columns:
        mse = mean_squared_error(df['total_expenses'], df['predicted_expense'])
        r2 = r2_score(df['total_expenses'], df['predicted_expense'])
        print(f"\n=== Regression Metrics ===")
        print(f"Mean Squared Error: {mse:,.2f}")
        print(f"R² Score: {r2:.2f}")
        
except Exception as e:
    print(f"❌ Error in regression prediction: {str(e)}")
    raise

print("\n🏦 Making classification predictions...")
try:
    # Ensure same feature order as training
    X_clf_df = pd.DataFrame(X_clf, columns=classification_features)
    
    # Predict financial health
    y_pred = clf_model.predict(X_clf_df)
    
    # Store predictions
    df['predicted_financial_health'] = label_encoder.inverse_transform(y_pred)
    
    if 'financial_health' in df.columns:
        print("\n=== Classification Report ===")
        print(classification_report(
            df['financial_health'],
            df['predicted_financial_health']
        ))
        
except Exception as e:
    print(f"❌ Error in classification prediction: {str(e)}")
    raise

# Save predictions
df.to_csv("data/final_predictions.csv", index=False)
print("\n✅ Predictions saved to data/final_predictions.csv")
output_file = "data/final_predictions.csv"
df.to_csv(output_file, index=False)
print(f"✅ Predictions saved to {output_file}\n")

# -----------------------------
# 8. Visualization setup
# -----------------------------
# Create output directories
plot_dir = os.path.join('static', 'images', 'plots')
os.makedirs(plot_dir, exist_ok=True)

# Only create user_plots directory if we're generating them
if not args.skip_user_plots:
    os.makedirs(os.path.join(plot_dir, 'user_plots'), exist_ok=True)

# Convert to datetime for plotting
if 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
else:
    # For the main dataset, we don't need to create months
    pass

users = df['user_id'].unique()

# -----------------------------
# 9. Generate visualizations for individual users (optional)
# -----------------------------
if not args.skip_user_plots:
    print("\n📊 Generating individual user plots...")
    
    # Load monthly data if it exists
    monthly_file = 'data/monthly_financial_data_large.csv'
    if os.path.exists(monthly_file):
        monthly_df = pd.read_csv(monthly_file)
        monthly_df['month'] = pd.to_datetime(monthly_df['month'])
        
        # Get unique users from monthly data
        users_to_plot = monthly_df['user_id'].unique()[:10]  # Limit to first 10 users for performance
        
        for user in users_to_plot:
            user_data = monthly_df[monthly_df['user_id'] == user].sort_values('month')
            if not user_data.empty:
                # Create user-specific directory
                user_plot_dir = os.path.join(plot_dir, 'user_plots', str(user))
                os.makedirs(user_plot_dir, exist_ok=True)
                
                # Generate visualizations for this user
                try:
                    # Example: Plot monthly income and expenses
                    plt.figure(figsize=(12, 6))
                    plt.plot(user_data['month'], user_data['total_income'], label='Income', marker='o')
                    plt.plot(user_data['month'], user_data['total_expenses'], label='Expenses', marker='x')
                    plt.title(f'Monthly Income vs Expenses - {user}')
                    plt.xlabel('Month')
                    plt.ylabel('Amount ($)')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(user_plot_dir, 'monthly_income_expenses.png'), dpi=150)
                    plt.close()
                    
                except Exception as e:
                    print(f"Error generating plots for user {user}: {str(e)}")
                    continue
        
        print(f"✅ Generated individual plots for {len(users_to_plot)} users.")
    else:
        print("⚠️ Monthly data not found. Skipping individual user plots.")
else:
    print("\n⏩ Skipping individual user plots generation (--skip-user-plots flag used)")

# -----------------------------
# 10. Generate Visualizations
# -----------------------------
print("📊 Generating visualizations...")

# 1. Methodology Visualizations
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

# Create distribution plots instead of time series
plt.figure(figsize=(14, 6))

# Plot 1: Distribution of Savings
plt.subplot(1, 2, 1)
sns.histplot(df['savings'], bins=30, kde=True, color='blue', alpha=0.5)
plt.title('Distribution of Savings')
plt.xlabel('Savings ($)')
plt.ylabel('Number of Users')
plt.grid(alpha=0.3)

# Plot 2: Distribution of Predicted Expenses
plt.subplot(1, 2, 2)
sns.histplot(df['predicted_expense'], bins=30, kde=True, color='orange', alpha=0.5)
plt.title('Distribution of Predicted Expenses')
plt.xlabel('Predicted Expense ($)')
plt.ylabel('Number of Users')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "savings_expense_distributions.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Distribution plots saved in '{plot_dir}' folder.\n")
print("🎉 All steps complete. Predictions and plots ready.")