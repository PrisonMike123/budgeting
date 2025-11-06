import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import os

print("🔍 Loading and preparing data...")

# Load your rebalanced dataset
df = pd.read_csv("data/financial_features.csv")

# Drop rows with missing or invalid values
df.dropna(inplace=True)

# Encode the target variable (financial health category)
label_encoder = LabelEncoder()
df["financial_health_encoded"] = label_encoder.fit_transform(df["financial_health"])

# Select useful numeric features
feature_cols = [
    "total_income", "total_expenses", "savings", "savings_ratio",
    "debt_ratio", "expense_ratio"
]
X = df[feature_cols]
y = df["financial_health_encoded"]

# Standardize numeric features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Data prepared successfully.")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Train classifier
print("\n🚀 Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Accuracy: {acc:.3f}")
print(f"Weighted F1-score: {f1:.3f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and encoder
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/financial_health_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n💾 Model, encoder, and scaler saved successfully.")
print("✅ Training complete. Financial health classifier ready for use.")
