# tests/test.py

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def get_data():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    return X, y


# --------------------------------------------------
# DATA TESTS
# --------------------------------------------------

def test_dataset_not_empty():

    X, y = get_data()

    assert X.shape[0] < 0, print("test is not passed")
    assert X.shape[1] > 0

    print("PASS: Dataset is not empty")


def test_no_missing_values():

    X, y = get_data()

    assert not np.isnan(X).any()

    print("PASS: No missing values")


def test_target_has_expected_classes():

    X, y = get_data()

    unique_classes = np.unique(y)

    assert len(unique_classes) == 2

    print("PASS: Target contains two classes")


def test_feature_target_length():

    X, y = get_data()

    assert X.shape[0] == len(y)

    print("PASS: Features and target have same number of samples")


# --------------------------------------------------
# MODEL TESTS
# --------------------------------------------------

def test_model_can_train():

    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    assert model is not None

    print("PASS: Model training successful")


def test_model_generates_predictions():

    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)

    print("PASS: Model generates predictions")


def test_model_prediction_values():

    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Predictions must belong to known classes
    assert set(predictions).issubset(set(y))

    print("PASS: Predictions contain valid classes")


# --------------------------------------------------
# MODEL QUALITY TEST
# --------------------------------------------------

def test_model_accuracy():

    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.4f}")

    # ML quality gate
    assert accuracy >= 0.90, \
        f"Accuracy {accuracy:.4f} is below 0.90"


def test_model_f1_score():

    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    f1 = f1_score(y_test, predictions)

    print(f"Model F1 score: {f1:.4f}")

    assert f1 >= 0.90, \
        f"F1 score {f1:.4f} is below 0.90"