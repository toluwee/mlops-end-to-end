import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from mlflow.models.signature import infer_signature
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_path: str, experiment_name: str = "iris-classification"):
    """Train model with MLflow tracking"""

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Load and prepare data
        data = pd.read_csv(data_path)
        X = data.drop(['target'], axis=1) if 'target' in data.columns else data.iloc[:, :-1]
        y = data['target'] if 'target' in data.columns else data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }

        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))

        # Create model signature
        signature = infer_signature(X_train, predictions)

        # Log model with updated parameter name
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",  # Using 'name' instead of 'artifact_path'
            signature=signature,
            registered_model_name="iris-classifier"
        )

        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, predictions)}")

        return model, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the iris dataset")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--experiment-name", type=str, default="iris-classification", help="MLflow experiment name")
    args = parser.parse_args()

    train_model(args.data_path, args.experiment_name)
