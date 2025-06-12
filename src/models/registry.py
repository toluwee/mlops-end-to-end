import argparse
import mlflow
from mlflow.tracking import MlflowClient

def promote_to_production():
    client = MlflowClient()
    model_name = "iris-classifier"

    # Get latest version
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise Exception(f"No versions found for model {model_name}")

    latest_version = max(versions, key=lambda x: int(x.version))
    version_number = latest_version.version

    # Remove production tag from all versions
    for version in versions:
        try:
            client.delete_model_version_tag(model_name, version.version, "production")
        except:
            pass

    # Tag latest version as production
    client.set_model_version_tag(
        name=model_name,
        version=version_number,
        key="production",
        value="true"
    )
    print(f"Successfully promoted version {version_number} to Production")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote-to-production", action="store_true", help="Promote latest model version to production")
    args = parser.parse_args()

    if args.promote_to_production:
        promote_to_production()
