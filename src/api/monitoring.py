import logging
import time
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Any
import threading

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model predictions and performance"""

    def __init__(self, window_size: int = 1000):
        """Initialize monitoring"""
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.feature_values = defaultdict(lambda: deque(maxlen=window_size))
        self.start_time = time.time()
        self.error_count = 0
        self.lock = threading.Lock()

    def log_prediction(self, features: List[float], prediction: int, latency: float = None):
        """Log a prediction"""
        with self.lock:
            self.predictions.append(prediction)
            if latency:
                self.latencies.append(latency)

            # Log feature values for drift detection
            for i, value in enumerate(features):
                self.feature_values[f"feature_{i}"].append(value)

    def log_error(self, error: str):
        """Log an error"""
        with self.lock:
            self.error_count += 1

    def detect_drift(self) -> Dict[str, Any]:
        """Simple drift detection based on feature statistics"""
        if len(self.features) < 100:
            return {"drift_detected": False, "reason": "Insufficient data"}

        recent_features = np.array(list(self.features)[-100:])
        older_features = np.array(list(self.features)[-200:-100]) if len(self.features) >= 200 else None

        if older_features is None:
            return {"drift_detected": False, "reason": "Insufficient historical data"}

        # Simple statistical test
        recent_mean = np.mean(recent_features, axis=0)
        older_mean = np.mean(older_features, axis=0)

        # Calculate relative change
        relative_change = np.abs((recent_mean - older_mean) / (older_mean + 1e-8))
        drift_threshold = 0.1  # 10% change threshold

        drift_detected = np.any(relative_change > drift_threshold)

        return {
            "drift_detected": drift_detected,
            "max_relative_change": float(np.max(relative_change)),
            "drift_threshold": drift_threshold,
            "features_with_drift": [i for i, change in enumerate(relative_change) if change > drift_threshold]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        if not self.predictions:
            return {
                "request_count": 0,
                "average_latency": 0.0,
                "error_count": self.error_count,
                "feature_statistics": {},
                "status": "No predictions made yet"
            }

        feature_stats = {}
        with self.lock:
            for feature_name, values in self.feature_values.items():
                if values:  # Only calculate stats if we have values
                    values_array = np.array(values)
                    feature_stats[feature_name] = {
                        "min": float(np.min(values_array)),
                        "max": float(np.max(values_array)),
                        "mean": float(np.mean(values_array)),
                        "std": float(np.std(values_array))
                    }

            return {
                "request_count": len(self.predictions),
                "average_latency": float(np.mean(self.latencies)) if self.latencies else 0.0,
                "error_count": self.error_count,
                "feature_statistics": feature_stats,
                "prediction_counts": dict(zip(*np.unique(list(self.predictions), return_counts=True))),
                "uptime_seconds": time.time() - self.start_time
            }
