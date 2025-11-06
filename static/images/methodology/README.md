# Methodology Visualizations

This directory contains the following methodology diagrams for the OakLedger project:

## 1. Data Preprocessing Flowchart
`data_preprocessing_flowchart.png`
- Visualizes the complete data preprocessing pipeline
- Shows the flow from raw data to train/test split
- Includes steps for data cleaning, feature engineering, and scaling

## 2. Random Forest Classifier Schematic
`random_forest_schematic.png`
- Illustrates the Random Forest architecture
- Shows input features, decision trees, and voting mechanism
- Demonstrates how the final prediction is made

## 3. Linear Regression Schematic
`linear_regression_schematic.png`
- Shows the concept of linear regression
- Displays actual data points and the best-fit line
- Includes the regression equation

## How to Update
To regenerate these diagrams:
1. Install required packages: `pip install graphviz matplotlib scikit-learn`
2. Run: `python create_methodology_diagrams.py`
3. The updated diagrams will be saved in this directory
