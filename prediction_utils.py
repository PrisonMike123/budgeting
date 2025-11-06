import pandas as pd
import joblib
import numpy as np

def load_models():
    """Load all the trained models and encoders."""
    return {
        'reg_model': joblib.load("models/linear_regression.pkl"),
        'clf_model': joblib.load("models/financial_health_model.pkl"),
        'scaler': joblib.load("models/scaler.pkl"),
        'label_encoder': joblib.load("models/label_encoder.pkl")
    }

def make_prediction(input_data):
    """
    Make predictions using the loaded models.
    
    Args:
        input_data (dict): Dictionary containing the input features
        
    Returns:
        dict: Dictionary containing the prediction results
    """
    # Load models
    models = load_models()
    
    # Prepare data for prediction
    features = {
        'age': float(input_data.get('age', 30)),
        'family_size': float(input_data.get('family_size', 1)),
        'total_income': float(input_data.get('total_income', 50000)),
        'total_expenses': float(input_data.get('total_expenses', 30000)),
        'savings': float(input_data.get('savings', 0)),
        'debt_ratio': float(input_data.get('debt_ratio', 0)),
        'expense_ratio': float(input_data.get('expense_ratio', 0.6)),
        'savings_ratio': float(input_data.get('savings_ratio', 0.2)),
        'credit_score': float(input_data.get('credit_score', 700))
    }
    
    # Prepare data for classification
    clf_features = [
        features['total_income'],
        features['total_expenses'],
        features['savings'],
        features['savings_ratio'],
        features['debt_ratio'],
        features['expense_ratio']
    ]
    
    # Scale the features
    X_clf = models['scaler'].transform([clf_features])
    
    # Make classification prediction
    health_pred = models['clf_model'].predict(X_clf)[0]
    health_label = models['label_encoder'].inverse_transform([health_pred])[0]
    
    # Prepare data for regression
    X_reg = [
        features['age'],
        features['family_size'],
        features['total_income'],
        features['debt_ratio'],
        features['expense_ratio'],
        features['savings_ratio'],
        features['credit_score']
    ]
    
    # Make regression prediction
    predicted_expense = models['reg_model'].predict([X_reg])[0]
    
    return {
        'financial_health': health_label,
        'predicted_expense': round(float(predicted_expense), 2),
        'savings_recommendation': f"Aim to save at least {max(20, min(40, 30 + (700 - features['credit_score']) // 20))}% of your income"
    }
