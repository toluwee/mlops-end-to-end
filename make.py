import subprocess
import sys

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def help():
    """Show help message"""
    print("\nUsage:")
    print("  python make.py <command>")
    print("\nAvailable commands:")
    print("  install      Install dependencies")
    print("  test         Run tests")
    print("  lint         Run linting")
    print("  format       Format code")
    print("  mlflow-server Start MLflow server")
    print("  train-model  Train model")
    print("  run-api      Run FastAPI server")
    print("  docker-build Build Docker images")
    print("  docker-run   Run with Docker Compose")
    print("  docker-stop  Stop Docker Compose")
    print("  k8s-create-namespace Create Kubernetes namespace")
    print("  k8s-deploy   Deploy to Kubernetes")
    print("  prometheus   Start Prometheus")
    print("  grafana      Start Grafana")
    print("  clean        Clean up temporary files")
    print("  data-sample  Generate sample data")
    print("  promote-model Promote latest model version to production")

def install():
    """Install dependencies"""
    run_command("pip install -r requirements/base.txt")
    run_command("pip install -r requirements/api.txt")
    run_command("pip install -r requirements/training.txt")

def test():
    """Run tests"""
    run_command("pytest tests/ -v --cov=src --cov-report=term-missing")

def lint():
    """Run linting"""
    run_command("flake8 src tests")
    run_command("black --check src tests")
    run_command("isort --check-only src tests")

def format_code():
    """Format code"""
    run_command("black src tests")
    run_command("isort src tests")

def mlflow_server():
    """Start MLflow server"""
    run_command("mlflow server --host 127.0.0.1 --port 5000")

def train_model():
    """Train model"""
    run_command("python -m src.models.train --data-path data/iris.csv")

def run_api():
    """Run FastAPI server"""
    run_command("uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000")

def docker_build():
    """Build Docker images"""
    run_command("docker build -f docker/Dockerfile.api -t mlops-api:latest .")

def docker_run():
    """Run with Docker Compose"""
    run_command("docker-compose -f docker/docker-compose.yml up -d")

def docker_stop():
    """Stop Docker Compose"""
    run_command("docker-compose -f docker/docker-compose.yml down")

def k8s_create_namespace():
    """Create Kubernetes namespace"""
    run_command("kubectl create namespace mlops --dry-run=client -o yaml | kubectl apply -f -")

def k8s_deploy():
    """Deploy to Kubernetes"""
    run_command("kubectl apply -f k8s/")

def prometheus():
    """Start Prometheus"""
    run_command("docker run -d -p 9090:9090 -v $(PWD)/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus")

def grafana():
    """Start Grafana"""
    run_command("docker run -d -p 3000:3000 grafana/grafana")

def clean():
    """Clean up temporary files"""
    run_command("find . -type f -name '*.pyc' -delete")
    run_command("find . -type d -name '__pycache__' -delete")
    run_command("find . -type d -name '*.egg-info' -exec rm -rf {} +")
    run_command("find . -type f -name '.coverage' -delete")

def data_sample():
    """Generate sample data"""
    run_command("python generate_sample_data.py")

def promote_model():
    """Promote latest model version to production"""
    run_command("python -m src.models.registry --promote-to-production")

COMMANDS = {
    "help": help,
    "install": install,
    "test": test,
    "lint": lint,
    "format": format_code,
    "mlflow-server": mlflow_server,
    "train-model": train_model,
    "run-api": run_api,
    "docker-build": docker_build,
    "docker-run": docker_run,
    "docker-stop": docker_stop,
    "k8s-create-namespace": k8s_create_namespace,
    "k8s-deploy": k8s_deploy,
    "prometheus": prometheus,
    "grafana": grafana,
    "clean": clean,
    "data-sample": data_sample,
    "promote-model": promote_model,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "help":
        help()
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        help()
        sys.exit(1)

    COMMANDS[command]()
