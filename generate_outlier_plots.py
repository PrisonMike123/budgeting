import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for plots if it doesn't exist
os.makedirs('outlier_plots', exist_ok=True)

# Load the cleaned data
expenses = pd.read_csv('expenses_cleaned.csv')

# Select numeric columns (excluding user_id)
numeric_cols = expenses.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'user_id' in numeric_cols:
    numeric_cols.remove('user_id')

# Function to create and save boxplot with highlighted outliers
def plot_outliers(data, column, save_path):
    plt.figure(figsize=(10, 6))
    
    # Calculate IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create boxplot
    ax = sns.boxplot(x=data[column])
    
    # Add title and labels
    plt.title(f'Outlier Analysis: {column}\nIQR Bounds: {lower_bound:,.2f} to {upper_bound:,.2f}')
    plt.xlabel('Amount (₹)')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Generate and save plots for each numeric column
for col in numeric_cols:
    save_path = f'outlier_plots/{col}_outliers.png'
    plot_outliers(expenses, col, save_path)
    print(f'Saved: {save_path}')

print("\n=== Outlier Analysis Complete ===")
print(f"Generated {len(numeric_cols)} boxplots in the 'outlier_plots' directory.")
print("These visualizations show the distribution of each expense category and highlight any outliers.")
print("Outliers are shown as points beyond the whiskers of the boxplots.")
