import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="iris-classifier",
    version=3,  # The version we just created
    stage="Production"
)
