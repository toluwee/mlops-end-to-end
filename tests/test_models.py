import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import sys
import os
import mlflow

# Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.evaluate import ModelEvaluator
from data.preprocessing import DataPreprocessor
from data.validation import DataValidator

@pytest.fixture
def iris_data():
    """Load iris dataset for testing"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y

def test_data_preprocessor(iris_data):
    """Test data preprocessing"""
    X, y = iris_data

    preprocessor = DataPreprocessor()
    X_scaled, y_encoded = preprocessor.fit_transform(X, y)

    assert X_scaled.shape == X.shape
    assert y_encoded.shape == y.shape
    assert preprocessor.is_fitted

    # Test transform
    X_test_scaled = preprocessor.transform(X.iloc[:10])
    assert X_test_scaled.shape == (10, X.shape[1])

def test_data_validator(iris_data):
    """Test data validation"""
    X, _ = iris_data

    validator = DataValidator(
        expected_features=X.columns.tolist(),
        feature_ranges={col: (X[col].min(), X[col].max()) for col in X.columns}
    )

    # Valid data
    result = validator.validate_input(X.iloc[:10])
    assert result["is_valid"]
    assert len(result["errors"]) == 0

    # Invalid data (wrong columns)
    invalid_data = X.copy()
    invalid_data.columns = ['wrong', 'column', 'names', 'here']
    result = validator.validate_input(invalid_data)
    assert not result["is_valid"]
    assert len(result["errors"]) > 0

def test_data_validator_feature_ranges(iris_data):
    """Test data validator with different feature ranges"""
    X, _ = iris_data

    # Test with tight ranges
    tight_ranges = {
        col: (X[col].mean() - 0.1, X[col].mean() + 0.1)
        for col in X.columns
    }
    validator = DataValidator(
        expected_features=X.columns.tolist(),
        feature_ranges=tight_ranges
    )

    # Should fail validation due to tight ranges
    result = validator.validate_input(X)
    assert not result["is_valid"]
    assert any("out of expected range" in err for err in result["errors"])

def test_data_validator_missing_features(iris_data):
    """Test data validator with missing features"""
    X, _ = iris_data

    # Create data with missing feature
    incomplete_data = X.drop(columns=[X.columns[0]])

    validator = DataValidator(
        expected_features=X.columns.tolist(),
        feature_ranges={col: (0, 10) for col in X.columns}
    )

    result = validator.validate_input(incomplete_data)
    assert not result["is_valid"]
    assert any("Missing expected feature" in err for err in result["errors"])

def test_data_validator_invalid_values(iris_data):
    """Test data validator with invalid values"""
    X, _ = iris_data

    # Create data with negative values
    invalid_data = X.copy()
    invalid_data.iloc[0, 0] = -1.0

    validator = DataValidator(
        expected_features=X.columns.tolist(),
        feature_ranges={col: (0, 10) for col in X.columns}
    )

    result = validator.validate_input(invalid_data)
    assert not result["is_valid"]
    assert any("out of expected range" in err for err in result["errors"])

def test_model_evaluator(iris_data):
    """Test model evaluation"""
    X, y = iris_data

    # Create dummy predictions
    y_pred = np.random.randint(0, 3, size=len(y))

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_classification(y.values, y_pred)

    assert "metrics" in results
    assert "accuracy" in results["metrics"]
    assert "confusion_matrix" in results
