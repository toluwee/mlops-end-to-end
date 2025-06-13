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
            self.predictions.append(int(prediction))  # Convert numpy.int32 to Python int
            if latency:
                self.latencies.append(float(latency))  # Convert to float if numpy type

            # Log feature values for drift detection
            for i, value in enumerate(features):
                self.feature_values[f"feature_{i}"].append(float(value))  # Convert to float if numpy type

    def log_error(self, error: str):
        """Log an error"""
        with self.lock:
            self.error_count += 1

    def detect_drift(self) -> Dict[str, Any]:
        """Simple drift detection based on feature statistics"""
        with self.lock:
            if len(next(iter(self.feature_values.values()), [])) < 100:
                return {"drift_detected": False, "reason": "Insufficient data"}

            drift_results = {}
            for feature_name, values in self.feature_values.items():
                values_array = np.array(list(values))
                recent = values_array[-100:]
                older = values_array[-200:-100] if len(values_array) >= 200 else None

                if older is None:
                    drift_results[feature_name] = {
                        "drift_detected": False,
                        "reason": "Insufficient historical data"
                    }
                    continue

                # Simple statistical test
                recent_mean = float(np.mean(recent))  # Convert to Python float
                older_mean = float(np.mean(older))  # Convert to Python float

                # Calculate relative change
                relative_change = abs((recent_mean - older_mean) / (older_mean + 1e-8))
                drift_threshold = 0.1  # 10% change threshold

                drift_results[feature_name] = {
                    "drift_detected": relative_change > drift_threshold,
                    "relative_change": float(relative_change),  # Convert to Python float
                    "threshold": drift_threshold
                }

            return {
                "feature_drift": drift_results,
                "any_drift_detected": any(r["drift_detected"] for r in drift_results.values())
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        with self.lock:
            if not self.predictions:
                return {
                    "request_count": 0,
                    "average_latency": 0.0,
                    "error_count": self.error_count,
                    "feature_statistics": {},
                    "status": "No predictions made yet"
                }

            predictions_list = list(self.predictions)
            feature_stats = {}
            for feature_name, values in self.feature_values.items():
                if values:  # Only calculate stats if we have values
                    values_array = np.array(list(values))
                    feature_stats[feature_name] = {
                        "min": float(np.min(values_array)),
                        "max": float(np.max(values_array)),
                        "mean": float(np.mean(values_array)),
                        "std": float(np.std(values_array))
                    }

            # Convert numpy types to Python native types
            prediction_counts = {}
            unique_predictions, counts = np.unique(predictions_list, return_counts=True)
            for pred, count in zip(unique_predictions, counts):
                prediction_counts[int(pred)] = int(count)

            return {
                "request_count": len(predictions_list),
                "average_latency": float(np.mean(list(self.latencies))) if self.latencies else 0.0,
                "error_count": self.error_count,
                "feature_statistics": feature_stats,
                "prediction_counts": prediction_counts,
                "uptime_seconds": float(time.time() - self.start_time)
            }
