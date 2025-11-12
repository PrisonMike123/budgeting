import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def create_financial_health_pie_chart(df, save_path='plots'):
    """Create a pie chart of financial health categories."""
    os.makedirs(save_path, exist_ok=True)
    
    # Count occurrences of each financial health category
    health_counts = df['financial_health'].value_counts()
    
    # Create color map (green for good, yellow for fair, red for poor)
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    patches, texts, autotexts = plt.pie(
        health_counts,
        labels=health_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.7, edgecolor='w'),
        textprops={'fontsize': 12}
    )
    
    # Make the text white and bold for better visibility
    for text in autotexts:
        text.set_color('white')
        text.set_fontweight('bold')
    
    plt.title('Distribution of Financial Health Categories', fontsize=16, pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Save the figure
    plt.savefig(f'{save_path}/financial_health_pie.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_income_savings_histograms(df, save_path='plots'):
    """Create histograms for total_income and savings_ratio."""
    os.makedirs(save_path, exist_ok=True)
    
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram for total_income
    sns.histplot(
        data=df, 
        x='total_income', 
        bins=20, 
        kde=True, 
        color='#4E79A7',
        edgecolor='white',
        ax=ax1
    )
    ax1.set_title('Distribution of Total Income', fontsize=14)
    ax1.set_xlabel('Total Income ($)')
    ax1.set_ylabel('Count')
    
    # Histogram for savings_ratio
    sns.histplot(
        data=df, 
        x='savings_ratio', 
        bins=20, 
        kde=True, 
        color='#59A14F',
        edgecolor='white',
        ax=ax2
    )
    ax2.set_title('Distribution of Savings Ratio', fontsize=14)
    ax2.set_xlabel('Savings Ratio')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/income_savings_histograms.png', dpi=300)
    plt.close()

def create_correlation_heatmap(df, save_path='plots'):
    """Create a correlation heatmap of numeric features."""
    os.makedirs(save_path, exist_ok=True)
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_data = df[numeric_cols].corr()
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list('custom', ['#2E86C1', '#F8F9F9', '#E74C3C'], N=100)
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))
    
    sns.heatmap(
        correlation_data,
        mask=mask,
        annot=True,
        cmap=cmap,
        center=0,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Correlation Heatmap of Numeric Features', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{save_path}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_heatmap(y_true, y_pred, classes, save_path='plots'):
    """Create a confusion matrix heatmap for the classifier."""
    from sklearn.metrics import confusion_matrix
    import itertools
    
    os.makedirs(save_path, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm_norm, 
        annot=True,
        cmap='Blues',
        fmt='.2f',
        xticklabels=classes,
        yticklabels=classes,
        cbar=False
    )
    
    # Add labels and title
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=16, pad=20)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_actual_vs_predicted_plot(y_true, y_pred, save_path='plots'):
    """Create a scatter plot of actual vs predicted values."""
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(
        y_true, 
        y_pred, 
        alpha=0.5,
        color='#2E86C1',
        edgecolor='white',
        linewidth=0.5
    )
    
    # Add reference line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Add labels and title
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Actual vs. Predicted Values', fontsize=16, pad=20)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{save_path}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_bar_chart(classification_report_dict, save_path='plots'):
    """Create a bar chart of precision, recall, and f1-score for each class."""
    os.makedirs(save_path, exist_ok=True)
    
    # Extract metrics for each class
    metrics = ['precision', 'recall', 'f1-score']
    classes = [cls for cls in classification_report_dict.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Prepare data for plotting
    data = {}
    for cls in classes:
        data[cls] = [classification_report_dict[cls][m] for m in metrics]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set width of bars
    bar_width = 0.25
    index = np.arange(len(classes))
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = [data[cls][i] for cls in classes]
        ax.bar(
            index + i * bar_width,
            values,
            bar_width,
            label=metric.capitalize(),
            alpha=0.8
        )
    
    # Add labels, title and legend
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics by Class', fontsize=16, pad=20)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='lower right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set y-axis limit
    plt.ylim(0, 1.1)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{save_path}/classification_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sample_predictions_table(df, n_samples=10, save_path='plots'):
    """Create a table of sample predictions with actual vs predicted values."""
    try:
        import matplotlib.font_manager as fm
        
        # Try to use DejaVu Sans, which has better Unicode support
        if 'DejaVu Sans' in fm.get_font_names():
            plt.rcParams['font.family'] = 'DejaVu Sans'
        elif 'Arial Unicode MS' in fm.get_font_names():
            plt.rcParams['font.family'] = 'Arial Unicode MS'
        
        os.makedirs(save_path, exist_ok=True)
        
        # Select sample rows
        sample_df = df.sample(min(n_samples, len(df)))
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Hide axes
        ax.axis('off')
        
        # Create table data - use text that's available in most fonts
        table_data = []
        for _, row in sample_df.iterrows():
            table_data.append([
                row.get('user_id', 'N/A'),
                f"${row.get('total_income', 0):,.2f}",
                f"${row.get('total_expenses', 0):,.2f}",
                f"${row.get('savings', 0):,.2f}",
                row.get('financial_health', 'N/A'),
                row.get('predicted_financial_health', 'N/A'),
                'CORRECT' if row.get('financial_health') == row.get('predicted_financial_health') else 'WRONG'
            ])
        
        # Create column headers
        col_labels = ['User ID', 'Income', 'Expenses', 'Savings', 'Actual', 'Predicted', 'Correct']
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colWidths=[0.15, 0.12, 0.12, 0.12, 0.15, 0.15, 0.1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(len(col_labels)):
            cell = table[0, i]
            cell.set_facecolor('#f2f2f2')
            cell.set_text_props(weight='bold')
        
        # Style data rows with alternating colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(col_labels)):
                cell = table[i, j]
                cell.set_facecolor('#ffffff' if i % 2 == 1 else '#f9f9f9')
        
        # Add title
        plt.title('Sample Predictions', fontsize=14, pad=20)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{save_path}/sample_predictions_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating sample predictions table: {str(e)}")
        raise

def create_methodology_visualizations(save_path='plots'):
    """Create visualizations for the methodology section using Matplotlib.
    
    This includes:
    1. Data Preprocessing Flowchart
    2. Random Forest Classifier Schematic
    3. Linear Regression Schematic
    """
    try:
        import os
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, BoxStyle
        from matplotlib.patches import Rectangle, FancyBboxPatch
        import numpy as np
        
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # -----------------------------
        # 1. Data Preprocessing Flowchart
        # -----------------------------
        print(" Creating data preprocessing flowchart...")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define box styles
        box_style = dict(boxstyle="round,pad=0.5", fc="lightblue", ec="steelblue", lw=2)
        start_end_style = dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="darkgreen", lw=2)
        
        # Define positions
        y_pos = [7, 6, 5, 4, 3, 2]
        
        # Draw boxes
        boxes = [
            ("Raw Financial Data", start_end_style),
            ("Handle Missing Values\n- Fill with mean/median\n- Drop if necessary", box_style),
            ("Feature Engineering\n- Calculate ratios\n- Create new features", box_style),
            ("Data Scaling\n- StandardScaler\n- Normalization", box_style),
            ("Data Splitting\n- Train/Test Split", box_style),
            ("Ready for Modeling", start_end_style)
        ]
        
        for i, (text, style) in enumerate(boxes):
            ax.text(0.5, y_pos[i], text, 
                   ha='center', va='center', 
                   bbox=style, fontsize=10)
            
            # Draw arrow
            if i < len(boxes) - 1:
                ax.arrow(0.5, y_pos[i] - 0.3, 0, -0.8, 
                        head_width=0.05, head_length=0.1, 
                        fc='black', ec='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 8)
        ax.axis('off')
        plt.title('Data Preprocessing Flowchart', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'data_preprocessing_flowchart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # -----------------------------
        # 2. Random Forest Classifier Schematic
        # -----------------------------
        print(" Creating Random Forest schematic...")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw decision trees
        tree_positions = [(0.2, 0.7), (0.5, 0.7), (0.8, 0.7)]
        for i, (x, y) in enumerate(tree_positions, 1):
            # Tree trunk
            ax.plot([x, x], [y, y - 0.2], 'k-', lw=2)
            # Tree top
            ax.add_patch(plt.Circle((x, y + 0.1), 0.15, color='lightgreen', ec='darkgreen', lw=2))
            ax.text(x, y + 0.1, f'Tree {i}', ha='center', va='center', fontsize=10)
        
        # Draw input features
        ax.text(0.5, 0.3, 'Input Features\n(Income, Expenses, etc.)', 
               ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="steelblue", lw=2))
        
        # Draw voting node
        ax.add_patch(plt.Circle((0.5, 0.5), 0.08, color='lightyellow', ec='goldenrod', lw=2))
        ax.text(0.5, 0.5, 'Vote', ha='center', va='center', fontsize=10)
        
        # Draw output
        ax.text(0.5, 0.1, 'Financial Health\nPrediction', 
               ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="steelblue", lw=2))
        
        # Draw arrows
        ax.arrow(0.5, 0.4, 0, 0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
        ax.arrow(0.5, 0.58, 0, 0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
        
        # Draw connections from trees to voting
        for x, _ in tree_positions:
            ax.arrow(x, 0.6, 0.5 - x, -0.1, 
                    head_width=0.02, head_length=0.02, 
                    fc='black', ec='black', linestyle='--', alpha=0.5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title('Random Forest Classifier', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'random_forest_schematic.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # -----------------------------
        # 3. Linear Regression Schematic
        # -----------------------------
        print(" Creating Linear Regression schematic...")
        plt.figure(figsize=(10, 6))
        
        # Generate sample data
        np.random.seed(42)
        X = np.linspace(0, 10, 100)
        y = 2 * X + 1 + np.random.normal(0, 1.5, 100)
        
        # Plot data points
        plt.scatter(X, y, color='#5C6BC0', alpha=0.6, label='Data Points')
        
        # Plot regression line
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        y_pred = model.predict(X.reshape(-1, 1))
        plt.plot(X, y_pred, color='#E53935', linewidth=2.5, 
                label=f'Regression Line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
        
        # Add error terms
        for i in range(0, 100, 10):
            plt.plot([X[i], X[i]], [y[i], y_pred[i]], 'k--', alpha=0.3)
        
        # Add labels and legend
        plt.title('Linear Regression Model', fontsize=16, pad=20)
        plt.xlabel('Feature (X)')
        plt.ylabel('Target (y)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'linear_regression_schematic.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" Methodology visualizations created successfully!")
        
    except Exception as e:
        print(f"Error creating methodology visualizations: {str(e)}")
        raise
