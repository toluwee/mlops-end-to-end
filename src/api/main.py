from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow.pyfunc
import pandas as pd
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
from contextlib import asynccontextmanager
import os
from sklearn.ensemble import RandomForestClassifier

from .models import PredictionRequest, PredictionResponse
from .monitoring import ModelMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment flags
IS_TESTING = os.getenv('TESTING', 'false').lower() == 'true'

# Prometheus metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total model requests')
REQUEST_DURATION = Histogram('model_request_duration_seconds', 'Model request duration')
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total predictions', ['predicted_class'])

# Global variables and model state
class ModelState:
    def __init__(self):
        self.model = None

state = ModelState()

def get_model():
    """Get current model instance"""
    return state.model

def set_model(model_instance):
    """Set current model instance"""
    state.model = model_instance

def create_test_model():
    """Create a simple test model for testing environment"""
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    # Train on a tiny dataset
    X = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [6.5, 3.0, 5.2, 2.0]]
    y = [0, 1, 2]
    clf.fit(X, y)
    return clf

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for model loading and cleanup"""
    try:
        # In testing mode, always use test model
        if IS_TESTING:
            logger.info("Testing mode detected, creating test model")
            test_model = create_test_model()
            wrapped_model = mlflow.pyfunc.PythonModel()
            wrapped_model._model_impl = test_model
            set_model(wrapped_model)
            logger.info("Created test model for testing environment")
        else:
            # Normal production mode
            client = mlflow.MlflowClient()
            model_name = "iris-classifier"

            # First try to load production model
            versions = client.search_model_versions(f"name='{model_name}' tags.production='true'")
            if versions:
                model_uri = f"models:/{model_name}/{versions[0].version}"
                set_model(mlflow.pyfunc.load_model(model_uri))
                logger.info(f"Loaded production model version {versions[0].version}")
            else:
                # Try any available version
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    model_uri = f"models:/{model_name}/{versions[0].version}"
                    set_model(mlflow.pyfunc.load_model(model_uri))
                    logger.info(f"Loaded test model version {versions[0].version}")
                else:
                    logger.warning("No MLflow models found")
                    raise Exception("No models available")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        if IS_TESTING:
            # Create test model as fallback in test mode
            test_model = create_test_model()
            wrapped_model = mlflow.pyfunc.PythonModel()
            wrapped_model._model_impl = test_model
            set_model(wrapped_model)
            logger.info("Created fallback test model")
        else:
            raise e  # Re-raise exception in production mode

    yield

    # Cleanup on shutdown
    set_model(None)
    logger.info("Model unloaded during shutdown")

app = FastAPI(
    title="MLOps Model API",
    description="Production-ready model serving API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": get_model() is not None,
        "timestamp": time.time()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions"""
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        model = get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert array to named features
        feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        input_dict = dict(zip(feature_names, request.features))
        input_data = pd.DataFrame([input_dict])

        # Make prediction
        if hasattr(model, '_model_impl'):
            prediction = model._model_impl.predict(input_data)
            probabilities = model._model_impl.predict_proba(input_data)
        else:
            prediction = model.predict(input_data)
            probabilities = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None

        # Log prediction for monitoring
        monitor.log_prediction(request.features, prediction[0], time.time() - start_time)

        # Update metrics
        PREDICTION_COUNTER.labels(predicted_class=str(prediction[0])).inc()

        # Ensure probabilities are included in response
        prob_list = probabilities[0].tolist() if probabilities is not None else None
        logger.info(f"Prediction probabilities: {prob_list}")

        response = PredictionResponse(
            prediction=int(prediction[0]),
            probability=prob_list,
            request_id=f"req_{int(time.time())}"
        )

        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)

        logger.info(f"Prediction made: {prediction[0]} (duration: {duration:.3f}s)")

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    client = mlflow.MlflowClient()
    model_name = "iris-classifier"
    versions = client.search_model_versions(f"name='{model_name}' tags.production='true'")
    version = versions[0] if versions else None

    return {
        "model_name": model_name,
        "version": version.version if version else None,
        "model_uri": f"models:/{model_name}/{version.version}" if version else None,
        "loaded_at": time.time(),
        "metadata": {
            "python_version": version.tags.get("python_version") if version else None,
            "model_type": "sklearn_classifier"
        }
    }

@app.get("/monitoring/statistics")
async def get_monitoring_statistics():
    """Get model monitoring statistics"""
    return monitor.get_statistics()
