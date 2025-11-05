import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load all CSV files
print("Loading data files...")
assets = pd.read_csv("oakledger_assets.csv")
expenses = pd.read_csv("oakledger_expenses.csv")
goals = pd.read_csv("oakledger_goals.csv")
income = pd.read_csv("oakledger_income.csv")
liabilities = pd.read_csv("oakledger_liabilities.csv")
transactions = pd.read_csv("oakledger_transactions.csv")
users = pd.read_csv("oakledger_users.csv")

# Function to display basic info about each dataframe
def display_basic_info(df, name):
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)

# Display basic info for all dataframes
dataframes = {
    'Assets': assets,
    'Expenses': expenses,
    'Goals': goals,
    'Income': income,
    'Liabilities': liabilities,
    'Transactions': transactions,
    'Users': users
}

for name, df in dataframes.items():
    display_basic_info(df, name)

# Handle missing values
print("\n=== Handling Missing Values ===")

# For expenses, fill missing values with median
expense_columns = expenses.select_dtypes(include=['float64', 'int64']).columns
for col in expense_columns:
    if expenses[col].isnull().sum() > 0:
        median_val = expenses[col].median()
        expenses[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in {col} with median: {median_val:.2f}")

# For transactions, ensure date is datetime
transactions['date'] = pd.to_datetime(transactions['date'])

# Handle outliers using IQR method
def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                print(f"\nOutliers in {col}:")
                print(f"- Lower bound: {lower_bound:.2f}")
                print(f"- Upper bound: {upper_bound:.2f}")
                print(f"- Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
                
                # Visualize outliers
                plt.figure(figsize=(8, 4))
                sns.boxplot(x=df[col])
                plt.title(f'Boxplot of {col}')
                plt.show()
                
                # Ask user if they want to remove outliers
                remove = input(f"Remove {len(outliers)} outliers from {col}? (y/n): ").lower()
                if remove == 'y':
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    print(f"Removed {len(df) - len(df_clean)} outliers from {col}")
    return df_clean

# Apply outlier treatment to numeric columns in expenses
expense_numeric_cols = expenses.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'user_id' in expense_numeric_cols:
    expense_numeric_cols.remove('user_id')

expenses_cleaned = remove_outliers(expenses, expense_numeric_cols)

# Save cleaned data
print("\nSaving cleaned data...")
expenses_cleaned.to_csv('expenses_cleaned.csv', index=False)
transactions.to_csv('transactions_cleaned.csv', index=False)

print("\n=== Data Cleaning Complete ===")
print(f"Original expenses shape: {expenses.shape}")
print(f"Cleaned expenses shape: {expenses_cleaned.shape}")

# Basic EDA
print("\n=== Basic EDA ===")

# 1. Summary statistics
print("\nExpenses Summary Statistics:")
print(expenses_cleaned.describe())

# 2. Correlation matrix
plt.figure(figsize=(10, 8))
corr = expenses_cleaned.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Expense Categories Correlation')
plt.tight_layout()
plt.savefig('expense_correlation.png')
plt.show()

# 3. Distribution of expenses
expense_cols = [col for col in expenses_cleaned.columns if 'expense' in col or 'spending' in col]
plt.figure(figsize=(12, 6))
for i, col in enumerate(expense_cols, 1):
    plt.subplot(2, 4, i)
    sns.histplot(expenses_cleaned[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('expense_distributions.png')
plt.show()

print("\n=== Data Analysis Complete ===")
print("Check the generated plots: expense_correlation.png and expense_distributions.png")
