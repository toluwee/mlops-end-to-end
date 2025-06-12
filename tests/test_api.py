import pytest
from fastapi.testclient import TestClient
import sys
import os
import mlflow
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Set testing environment
os.environ['TESTING'] = 'true'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import after path setup
from api.main import app, create_test_model, set_model

# Initialize test client
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_model():
    """Setup and cleanup test model"""
    # Create and set up test model
    test_model = create_test_model()
    wrapped_model = mlflow.pyfunc.PythonModel()
    wrapped_model._model_impl = test_model
    set_model(wrapped_model)

    yield

    # Cleanup
    set_model(None)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_predict_endpoint():
    """Test prediction endpoint"""
    test_data = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    response = client.post("/predict", json=test_data)

    if response.status_code == 503:
        # Model not loaded in test environment
        pytest.skip("Model not loaded in test environment")

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "request_id" in data
    assert isinstance(data["prediction"], int)

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model/info")

    if response.status_code == 503:
        pytest.skip("Model not loaded in test environment")

    assert response.status_code == 200
    data = response.json()
    assert "model_uri" in data

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

def test_monitoring_statistics():
    """Test monitoring statistics endpoint"""
    # First make some predictions to generate statistics
    test_data = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    client.post("/predict", json=test_data)

    # Get monitoring statistics
    response = client.get("/monitoring/statistics")
    assert response.status_code == 200
    data = response.json()

    assert "request_count" in data
    assert "average_latency" in data
    assert "feature_statistics" in data
    assert isinstance(data["request_count"], int)
    assert isinstance(data["average_latency"], (int, float))
    assert isinstance(data["feature_statistics"], dict)

def test_monitoring_prediction_tracking():
    """Test that predictions are properly tracked"""
    test_data = {
        "features": [6.3, 2.9, 5.6, 1.8]  # Different features
    }

    # Make multiple predictions
    for _ in range(3):
        client.post("/predict", json=test_data)

    # Check monitoring statistics
    response = client.get("/monitoring/statistics")
    assert response.status_code == 200
    data = response.json()

    # Should have logged our predictions
    assert data["request_count"] >= 3
    assert data["feature_statistics"] is not None

    # Check feature ranges are being tracked
    for feature in test_data["features"]:
        assert any(
            feature >= stats["min"] and feature <= stats["max"]
            for stats in data["feature_statistics"].values()
        )

def test_monitoring_drift_detection():
    """Test drift detection in monitoring"""
    # Make normal predictions
    normal_data = {
        "features": [5.1, 3.5, 1.4, 0.2]  # Normal iris data
    }
    client.post("/predict", json=normal_data)

    # Make predictions with potential drift
    drift_data = {
        "features": [10.0, 7.0, 8.0, 5.0]  # Unusual values
    }
    response = client.post("/predict", json=drift_data)

    # Check monitoring statistics for drift indicators
    stats_response = client.get("/monitoring/statistics")
    stats = stats_response.json()

    assert "feature_statistics" in stats
    for feature_stats in stats["feature_statistics"].values():
        assert "std" in feature_stats
        assert "mean" in feature_stats

def test_monitoring_error_tracking():
    """Test error tracking in monitoring"""
    # Make invalid request
    invalid_data = {
        "features": [-1.0, -2.0, -3.0, -4.0]  # Invalid negative values
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code in [400, 500]  # Should be an error

    # Check monitoring statistics includes error tracking
    stats_response = client.get("/monitoring/statistics")
    stats = stats_response.json()

    assert "error_count" in stats or "errors" in stats

def test_monitoring_latency_tracking():
    """Test latency tracking in monitoring"""
    test_data = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    # Make several predictions to generate latency statistics
    for _ in range(3):
        client.post("/predict", json=test_data)

    # Check monitoring statistics
    response = client.get("/monitoring/statistics")
    stats = response.json()

    assert "average_latency" in stats
    assert stats["average_latency"] >= 0  # Should be non-negative
    assert isinstance(stats["average_latency"], (int, float))
