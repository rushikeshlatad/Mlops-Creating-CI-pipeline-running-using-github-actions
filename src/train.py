# src/train.py

import os
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


MODEL_PATH = "model.joblib"

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# --------------------------------------------------
# 2. DATA VALIDATION GATE
# --------------------------------------------------

print("\nRunning data validation...")

# Check dataset is not empty
assert X.shape[0] > 0, "Dataset contains no samples"

# Check number of features
assert X.shape[1] > 0, "Dataset contains no features"

# Check missing values
assert not __import__("numpy").isnan(X).any(), \
    "Dataset contains missing values"

# Check target exists
assert len(y) > 0, "Target variable is empty"

# Check binary classification
assert len(set(y)) == 2, \
    "Expected binary classification problem"

print("Data validation PASSED")


# --------------------------------------------------
# 3. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Train Model
# --------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed")


# --------------------------------------------------
# 5. Model Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance")
print("-----------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# --------------------------------------------------
# 6. MODEL QUALITY GATE
# --------------------------------------------------

MIN_ACCURACY = 0.90
MIN_F1 = 0.90

print("\nRunning model quality gate...")

assert accuracy >= MIN_ACCURACY, (
    f"Model accuracy {accuracy:.4f} is below "
    f"required threshold {MIN_ACCURACY}"
)

assert f1 >= MIN_F1, (
    f"Model F1 score {f1:.4f} is below "
    f"required threshold {MIN_F1}"
)

print("Model quality gate PASSED")


# --------------------------------------------------
# 7. Save Model
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")