import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrow, FancyArrowPatch
from matplotlib.patches import BoxStyle
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as path_effects

def create_preprocessing_flowchart():
    """Create a visually appealing flowchart for data preprocessing steps."""
    # Set up the figure with a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Professional color palette
    colors = [
        '#4285F4',  # Blue
        '#34A853',  # Green
        '#FBBC05',  # Yellow
        '#EA4335',  # Red
        '#673AB7'   # Purple
    ]
    
    # Define node properties with better styling
    nodes = [
        {'id': 'A', 
         'label': 'Raw Data\n(monthly_financial_data.csv)', 
         'pos': (0.1, 0.5), 
         'color': colors[0],
         'text_color': 'white'},
         
        {'id': 'B', 
         'label': 'Data Cleaning\n• Handle missing values\n• Remove duplicates\n• Fix data types', 
         'pos': (0.3, 0.5), 
         'color': colors[1],
         'text_color': 'white'},
         
        {'id': 'C', 
         'label': 'Feature Engineering\n• Calculate ratios\n• Create new features\n• Encode categories', 
         'pos': (0.5, 0.5), 
         'color': colors[2],
         'text_color': '#202124'},
         
        {'id': 'D', 
         'label': 'Data Scaling\n• Standardize numerical\n• Normalize ranges', 
         'pos': (0.7, 0.5), 
         'color': colors[3],
         'text_color': 'white'},
         
        {'id': 'E', 
         'label': 'Train/Test Split\n• 80/20 split\n• Stratified sampling', 
         'pos': (0.9, 0.5), 
         'color': colors[4],
         'text_color': 'white'}
    ]
    
    # Draw edges with gradient effect
    for i in range(len(nodes)-1):
        ax.annotate('', 
                   xy=(nodes[i+1]['pos'][0]-0.12, nodes[i+1]['pos'][1]), 
                   xycoords='axes fraction',
                   xytext=(nodes[i]['pos'][0]+0.12, nodes[i]['pos'][1]), 
                   textcoords='axes fraction',
                   arrowprops=dict(
                       arrowstyle='fancy',
                       color=nodes[i]['color'],
                       connectionstyle=f'arc3,rad=0.1',
                       shrinkA=15, 
                       shrinkB=15,
                       linewidth=2.5,
                       alpha=0.8
                   ))
    
    # Draw nodes with shadow effect
    for node in nodes:
        # Add shadow
        shadow = patches.FancyBboxPatch(
            (node['pos'][0]-0.12 + 0.01, node['pos'][1]-0.15 - 0.01), 
            0.24, 0.3,
            boxstyle=patches.BoxStyle("Round", pad=0.1, rounding_size=0.1),
            facecolor='#555555', 
            edgecolor='none',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(shadow)
        
        # Draw node
        rect = patches.FancyBboxPatch(
            (node['pos'][0]-0.12, node['pos'][1]-0.15), 
            0.24, 0.3,
            boxstyle=patches.BoxStyle("Round", pad=0.1, rounding_size=0.1),
            facecolor=node['color'], 
            edgecolor='white', 
            linewidth=2.5,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add text with better typography
        for j, line in enumerate(node['label'].split('\n')):
            weight = 'bold' if j == 0 else 'normal'
            size = 10 if j == 0 else 9
            plt.text(
                node['pos'][0], 
                node['pos'][1] - 0.05 + (0.1 - (0.1 * j)),
                line,
                ha='center', 
                va='center', 
                fontsize=size, 
                fontweight=weight,
                color=node['text_color'],
                zorder=3
            )
    
    # Add title with better styling
    plt.title(
        'Data Preprocessing Pipeline', 
        fontsize=20, 
        fontweight='bold', 
        pad=30,
        color='#2D3748'
    )
    
    # Add subtle background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')
    
    # Remove axis and set tight layout
    plt.axis('off')
    plt.tight_layout()
    
    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#E2E8F0')
        spine.set_linewidth(1.5)
    
    # Save the diagram
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'data_preprocessing_flowchart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_rf_schematic():
    """Create an enhanced Random Forest Classifier schematic."""
    # Set up the figure with a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color palette
    colors = {
        'features': '#4285F4',
        'trees': ['#34A853', '#FBBC05', '#EA4335'],
        'voting': '#673AB7',
        'output': '#202124'
    }
    
    # Title
    plt.title('Random Forest Classifier Architecture', 
             fontsize=20, 
             fontweight='bold', 
             pad=30,
             color='#2D3748')
    
    # Input features section
    plt.text(0.1, 0.85, 'Input Features', 
            fontsize=14, 
            fontweight='bold', 
            ha='center',
            color='#2D3748')
    
    features = [
        'Monthly Income', 
        'Savings', 
        'Debt', 
        'Expenses', 
        'Credit Score'
    ]
    
    # Draw feature boxes with shadow
    for i, feature in enumerate(features):
        # Shadow
        shadow = patches.Rectangle(
            (0.05 + 0.005, 0.7 - i*0.12 - 0.005), 
            0.15, 0.1,
            color='#555555', 
            alpha=0.2,
            zorder=1
        )
        ax.add_patch(shadow)
        
        # Feature box
        rect = patches.Rectangle(
            (0.05, 0.7 - i*0.12), 
            0.15, 0.1,
            facecolor=colors['features'],
            edgecolor='white',
            linewidth=1.5,
            zorder=2,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Feature text
        plt.text(0.125, 0.75 - i*0.12, feature, 
                ha='center', 
                va='center', 
                fontsize=10,
                color='white',
                fontweight='bold',
                zorder=3)
    
    # Decision trees with better styling
    tree_positions = [0.3, 0.5, 0.7]
    
    for i, pos in enumerate(tree_positions):
        # Tree trunk
        plt.plot([pos, pos], [0.5, 0.35], 
                color='#8D6E63', 
                linewidth=3,
                zorder=2)
        
        # Tree leaves (circles)
        for j in range(3):
            y_pos = 0.5 + (j+1)*0.12
            circle = patches.Circle(
                (pos, y_pos), 
                0.08, 
                facecolor=colors['trees'][i],
                edgecolor='white',
                linewidth=1.5,
                zorder=3,
                alpha=0.9
            )
            ax.add_patch(circle)
            
            # Add tree label
            plt.text(pos, y_pos, f'DT{i+1}', 
                    ha='center', 
                    va='center', 
                    fontsize=10,
                    fontweight='bold',
                    color='white',
                    zorder=4)
    
    # Voting mechanism with better styling
    voting_circle = patches.Circle(
        (0.85, 0.5), 
        0.1, 
        facecolor=colors['voting'],
        edgecolor='white',
        linewidth=2,
        zorder=3,
        alpha=0.9
    )
    ax.add_patch(voting_circle)
    
    plt.text(0.85, 0.5, 'Voting\nMechanism', 
            ha='center', 
            va='center', 
            fontsize=10,
            fontweight='bold',
            color='white',
            zorder=4)
    
    # Output with better styling
    output_rect = patches.FancyBboxPatch(
        (0.98, 0.45), 
        0.12, 0.1,
        boxstyle=patches.BoxStyle("Round", pad=0.1, rounding_size=0.1),
        facecolor=colors['output'],
        edgecolor='white',
        linewidth=2,
        zorder=3,
        alpha=0.9
    )
    ax.add_patch(output_rect)
    
    plt.text(1.04, 0.5, 'Financial\nHealth\nScore', 
            ha='center', 
            va='center', 
            fontsize=10,
            fontweight='bold',
            color='white',
            zorder=4)
    
    # Arrows with better styling
    for i in range(5):
        # From features to trees
        ax.arrow(0.2, 0.75 - i*0.12, 0.08, -0.2 + i*0.03, 
                head_width=0.02, 
                head_length=0.02, 
                fc=colors['features'], 
                ec=colors['features'],
                length_includes_head=True,
                width=0.005,
                alpha=0.7)
        
        # From trees to voting
        if i < 3:
            ax.arrow(tree_positions[i] + 0.1, 0.5, 0.75 - tree_positions[i] - 0.1, 0,
                   head_width=0.02, 
                   head_length=0.02, 
                   fc=colors['trees'][i], 
                   ec=colors['trees'][i],
                   length_includes_head=True,
                   width=0.005,
                   alpha=0.7)
    
    # From voting to output
    ax.arrow(0.95, 0.5, 0.02, 0, 
            head_width=0.02, 
            head_length=0.02, 
            fc=colors['voting'], 
            ec=colors['voting'],
            length_includes_head=True,
            width=0.005,
            alpha=0.7)
    
    # Add subtle background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')
    
    # Remove axis and set tight layout
    plt.axis('off')
    plt.tight_layout()
    
    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#E2E8F0')
        spine.set_linewidth(1.5)
    
    # Save the diagram
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'random_forest_schematic.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_linear_regression_schematic():
    """Create an enhanced linear regression schematic."""
    # Set up the figure with a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate some sample data with more points for better visualization
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    y = 2 * X + 1 + np.random.normal(0, 1.5, 30)
    
    # Calculate regression line
    m, b = np.polyfit(X, y, 1)
    
    # Create a grid for the background
    ax.set_axisbelow(True)
    
    # Plot data points with better styling
    plt.scatter(X, y, 
               color='#4285F4', 
               s=100,
               alpha=0.8,
               edgecolors='white',
               linewidth=1.5,
               zorder=3,
               label='Data Points')
    
    # Plot regression line with confidence interval
    plt.plot(X, m*X + b, 
            color='#EA4335', 
            linewidth=3,
            alpha=0.9,
            zorder=2,
            label='Regression Line')
    
    # Add equation with better styling
    equation = f'y = {m:.2f}x + {b:.2f}'
    r_squared = np.corrcoef(X, y)[0, 1] ** 2
    
    # Add a nice text box
    text_box = patches.FancyBboxPatch(
        (0.05, 0.85), 0.3, 0.12,
        boxstyle=patches.BoxStyle("Round", pad=0.1, rounding_size=0.1),
        facecolor='white',
        edgecolor='#E2E8F0',
        linewidth=1.5,
        alpha=0.9,
        zorder=4
    )
    ax.add_patch(text_box)
    
    plt.text(0.2, 0.9, 'Regression Equation', 
            ha='left', va='center', 
            fontsize=11, 
            fontweight='bold',
            color='#2D3748',
            zorder=5)
    
    plt.text(0.2, 0.85, equation + f'\nR² = {r_squared:.3f}', 
            ha='left', va='center', 
            fontsize=10, 
            color='#4A5568',
            zorder=5)
    
    # Add labels and title with better styling
    plt.title('Linear Regression Model', 
             fontsize=20, 
             fontweight='bold', 
             pad=20,
             color='#2D3748')
    
    plt.xlabel('Input Feature (e.g., Monthly Income)', 
              fontsize=12, 
              labelpad=10,
              color='#4A5568')
    
    plt.ylabel('Predicted Value (e.g., Expenses)', 
              fontsize=12, 
              labelpad=10,
              color='#4A5568')
    
    # Customize grid and ticks
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tick_params(axis='both', which='both', length=0)
    
    # Add legend with better styling
    legend = plt.legend(loc='upper left', 
                       bbox_to_anchor=(0.7, 0.95),
                       frameon=True,
                       framealpha=0.9,
                       edgecolor='#E2E8F0',
                       facecolor='white')
    
    # Add subtle background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#E2E8F0')
        spine.set_linewidth(1.5)
    
    # Add a subtle shadow effect to the plot area
    for i in range(5):
        shadow = plt.Rectangle((0.01 - i*0.001, 0.01 - i*0.001), 
                             0.98 + i*0.002, 0.98 + i*0.002,
                             transform=ax.transAxes,
                             fill=None,
                             edgecolor='#E2E8F0',
                             alpha=0.1,
                             zorder=0)
        ax.add_patch(shadow)
    
    plt.tight_layout()
    
    # Save the diagram
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'linear_regression_schematic.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('static/images/plots', exist_ok=True)
    
    # Set a consistent style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    try:
        # Generate all diagrams
        print("Creating Data Preprocessing Flowchart...")
        create_preprocessing_flowchart()
        
        print("Creating Random Forest Classifier Schematic...")
        create_rf_schematic()
        
        print("Creating Linear Regression Schematic...")
        create_linear_regression_schematic()
        
        print("\n✅ All methodology diagrams have been successfully created in 'static/images/methodology/'")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please ensure all required Python packages are installed.")
        print("You can install them using: pip install matplotlib numpy scikit-learn")
