import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, compute_model_metrics



def test_process_data_returns_expected_rows():
    """
    Test that process_data returns feature and label arrays
    with the expected number of rows.
    """
    data = pd.DataFrame({
        "age": [39, 50],
        "workclass": ["State-gov", "Self-emp-not-inc"],
        "fnlgt": [77516, 83311],
        "education": ["Bachelors", "Bachelors"],
        "education-num": [13, 13],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [2174, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })

    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == 2
    assert len(y) == 2
    assert encoder is not None
    assert lb is not None




def test_train_model_returns_random_forest():
    """
    Test that train_model returns a fitted RandomForestClassifier.
    """
    X_train = [[0, 1], [1, 0], [1, 1], [0, 0]]
    y_train = [0, 1, 1, 0]

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)



def test_compute_model_metrics_range():
    """
    Test that precision, recall, and fbeta are between 0 and 1.
    """
    y = [1, 0, 1, 1]
    preds = [1, 0, 0, 1]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
