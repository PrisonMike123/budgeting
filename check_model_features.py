import joblib
import pandas as pd

def print_model_info(model, model_name):
    print(f"\n=== {model_name} ===")
    try:
        print(f"Number of features expected: {model.n_features_in_}")
        print(f"Feature names: {getattr(model, 'feature_names_in_', 'Not available')}")
        if hasattr(model, 'feature_importances_'):
            print("Feature importances:")
            for name, importance in zip(model.feature_names_in_, model.feature_importances_):
                print(f"  {name}: {importance:.4f}")
        elif hasattr(model, 'coef_'):
            print(f"Model coefficients shape: {model.coef_.shape}")
            if hasattr(model, 'feature_names_in_'):
                for name, coef in zip(model.feature_names_in_, model.coef_[0]):
                    print(f"  {name}: {coef:.4f}")
    except Exception as e:
        print(f"Could not get feature information: {e}")

def main():
    try:
        # Load the models
        print("Loading models...")
        models = {
            'Linear Regression': joblib.load("models/linear_regression.pkl"),
            'Random Forest Classifier': joblib.load("models/financial_health_model.pkl"),
            'Scaler': joblib.load("models/scaler.pkl")
        }
        
        # Print model information
        for name, model in models.items():
            print_model_info(model, name)
            
        # Print available data columns
        print("\n=== Data Information ===")
        data = pd.read_csv("data/financial_features.csv")
        print("Available columns in data:", list(data.columns))
        print("\nFirst few rows of data:")
        print(data.head())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
