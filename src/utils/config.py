import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_config() -> Dict[str, Any]:
    """Get application configuration"""
    return {
        "mlflow": {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            "model_name": os.getenv("MODEL_NAME", "iris-classifier"),
            "model_stage": os.getenv("MODEL_STAGE", "Production")
        },
        "api": {
            "host": os.getenv("API_HOST", "0.0.0.0"),
            "port": int(os.getenv("API_PORT", "8000")),
            "workers": int(os.getenv("API_WORKERS", "4"))
        },
        "monitoring": {
            "window_size": int(os.getenv("MONITORING_WINDOW_SIZE", "1000")),
            "drift_threshold": float(os.getenv("DRIFT_THRESHOLD", "0.1"))
        }
    }
