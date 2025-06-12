import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from typing import Dict, Any, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Comprehensive classification evaluation"""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.close()

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)

        # Log detailed report
        mlflow.log_dict(report, "classification_report.json")

        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")

        return {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
