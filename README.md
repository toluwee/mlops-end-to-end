# Complete MLOps Pipeline

A production-ready MLOps pipeline implementing best practices for model training, serving, monitoring, and deployment using modern tools and technologies.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Badges](#badges)
- [Environment Variables](#environment-variables)
- [Quick Start](#quick-start)
- [Running the Pipeline](#running-the-pipeline)
- [Monitoring](#monitoring)
- [API Usage](#api-usage)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Science  │    │   MLOps Core    │    │   Production    │
│                 │    │                 │    │                 │
│ • Jupyter       │    │ • MLflow        │    │ • Kubernetes    │
│ • Experiments   │───▶│ • Model Registry│───▶│ • Load Balancer │
│ • Feature Eng   │    │ • Artifacts     │    │ • Auto-scaling  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │   Integration   │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • FastAPI       │    │ • Docker        │    │ • Prometheus    │
│ • Testing       │    │ • CI/CD         │    │ • Grafana       │
│ • Validation    │    │ • Registry      │    │ • Alerting      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

- **Model Training & Tracking**: MLflow for experiment tracking and model registry
- **API Serving**: FastAPI with automatic documentation and validation
- **Containerization**: Docker and Docker Compose for consistent environments
- **Orchestration**: Kubernetes deployment with auto-scaling
- **Monitoring**: Prometheus metrics, Grafana dashboards, and drift detection
- **Logging**: Structured JSON logging
- **Data Validation**: Input validation and feature drift monitoring

## Badges

<!-- Add badges as needed, e.g. build, coverage, license -->

## Environment Variables

All configuration is managed via environment variables. Copy `.env.example` to `.env` and adjust as needed.

| Variable                      | Description                                 | Default/Example           |
|-------------------------------|---------------------------------------------|---------------------------|
| MLFLOW_TRACKING_URI           | MLflow server URI                           | http://127.0.0.1:5000     |
| MLFLOW_EXPERIMENT_NAME        | MLflow experiment name                      | iris-classification       |
| MLFLOW_REGISTERED_MODEL_NAME  | MLflow registered model name                | iris-classifier           |
| API_HOST                      | API server host                             | 0.0.0.0                  |
| API_PORT                      | API server port                             | 8000                     |
| LOG_LEVEL                     | Logging level                               | INFO                     |
| PROMETHEUS_PORT               | Prometheus port                             | 9090                     |
| GRAFANA_PORT                  | Grafana port                                | 3000                     |
| DB_HOST                       | Database host (if using external DB)        | localhost                 |
| DB_PORT                       | Database port                               | 5432                     |
| DB_NAME                       | Database name                               | mlflow                   |
| DB_USER                       | Database user                               | mlflow                   |
| DB_PASSWORD                   | Database password                           | mlflow_password           |

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (optional, for containerized deployment)
- Kubernetes cluster (optional, for production deployment)

### Installation

1. **Clone and Setup**
   ```bash
   git clone https://github.com/your-username/mlops-complete-pipeline.git
   cd mlops-complete-pipeline
   python -m venv venv
   
   # Activate virtual environment
   # For Windows:
   venv\Scripts\activate
   # For Unix/MacOS:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   # Works on all platforms (Windows/Unix/MacOS):
   python make.py install
   ```

## Running the Pipeline
### Option 1: Local Development
All commands below work the same way on Windows, Unix, and MacOS:

1. **Start MLflow Server**
   ```bash
   python make.py mlflow-server
   ```

2. **Train Initial Model**
   ```bash
   # Open another terminal --be sure to activate the virtual environment there as well
   python make.py train-model
   ```
   View training results at http://127.0.0.1:5000 (Windows)  http://localhost:5000 (unix/macOS)


3. **Promote Preferred Model**
   ```bash
   # Once a model with preferred performance is identified, promote it.
   # TODO: implement automatic promotion of best model based on a metric threshold
   python make.py promote-model
   ```



4. **Start API Server**
   ```bash
   # stop and restart the mlflow server before this to ensure model has been promoted
   python make.py run-api
   
   
   
   ```

### Endpoints

You can access the API at http://127.0.0.1:8000 with the following available endpoints:

- `/docs` - Interactive API documentation (Swagger UI)
- `/metrics` - Prometheus metrics
- `/health` - Health check endpoint
- `/monitoring/statistics` - Model monitoring statistics
- `/predict` - Prediction endpoint (POST requests)

For instance, API documentation is available at http://127.0.0.1:8000/docs

### Option 2:  Docker Deployment

Commands are the same for all platforms:
```bash
# Start all services
python make.py docker-run

# Stop all services
python make.py docker-stop
```

This will start:
- MLflow server (http://127.0.0.1:5000)
- Model API (http://127.0.0.1:8000)
- Prometheus (http://127.0.0.1:9090)
- Grafana (http://127.0.0.1:3000)

### Option 3:  Kubernetes Deployment

1. **Create Namespace**
   ```bash
   python make.py k8s-create-namespace
   ```

2. **Deploy Application**
   ```bash
   python make.py k8s-deploy
   ```

## Monitoring

### Available Metrics
- Model prediction latency
- Request counts by prediction class
- Feature drift detection
- System metrics (CPU, memory, etc.)

### Endpoints
- `/metrics` - Prometheus metrics
- `/health` - API health check
- `/monitoring/statistics` - Model monitoring statistics
- `/model/info` - Current model information

### Grafana Dashboards
Access the monitoring dashboard at http://127.0.0.1:3000 (default credentials: admin/admin)

## API Usage

### Health Check
```bash
# Windows
curl http://127.0.0.1:8000/health

# Unix/MacOS
curl http://localhost:8000/health
```

### Make Prediction

1. Using curl from the command line:
```bash
# Windows
Using Command Prompt:

curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"features\": [5.1, 3.5, 1.4, 0.2]}"


# Unix/MacOS
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```
2. Using PowerShell:
```powershell
$body = @{
    features = @(5.1, 3.5, 1.4, 0.2)
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -Headers @{"Content-Type"="application/json"} `
    -Body $body
```

3 Using Python:
```python
import requests

data = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=data
)
print(response.json())
```

4. Using the Swagger UI interface:
- Open http://127.0.0.1:8000/docs in your browser
- Find the `/predict` endpoint
- Click on "Try it out"
- Enter your input in the JSON format:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```
- Click "Execute"

The features array represents Iris flower measurements in this order:
- Sepal Length: 5.1
- Sepal Width: 3.5
- Petal Length: 1.4
- Petal Width: 0.2

### View Metrics
```bash
# Windows
curl http://127.0.0.1:8000/metrics

# Unix/MacOS
curl http://localhost:8000/metrics
```

### View Model Statistics
```bash
# Windows
curl http://127.0.0.1:8000/monitoring/statistics

# Unix/MacOS
curl http://localhost:8000/monitoring/statistics
```

### API Documentation
Visit `http://127.0.0.1:8000/docs` (Windows) or `http://localhost:8000/docs` (Unix/MacOS) for interactive API documentation.

## Monitoring

- **MLflow UI**: http://127.0.0.1:5000 (Windows) or http://localhost:5000 (Unix/MacOS)
- **Prometheus**: http://127.0.0.1:9090 (Windows) or http://localhost:9090 (Unix/MacOS)
- **Grafana**: http://127.0.0.1:3000 (Windows) or http://localhost:3000 (Unix/MacOS) (admin/admin)

## Testing

```bash
python make.py test
```

## Code Quality

```bash
python make.py lint
python make.py format
```

## Troubleshooting & FAQ

- **Virtual environment not activating?**
  - On Windows, use `venv\Scripts\activate`. On Unix/MacOS, use `source venv/bin/activate`.
- **MLflow not tracking runs?**
  - Ensure `MLFLOW_TRACKING_URI` is set correctly in your `.env` file.
- **Docker build fails?**
  - Make sure Docker is running and you have permissions.
- **Kubernetes deployment issues?**
  - Check your cluster context and namespace.

## References
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)

## Project Structure

```
mlops-complete-pipeline/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   │   ├── main.py       # API endpoints
│   │   ├── models.py     # Data models
│   │   └── monitoring.py # Monitoring setup
│   ├── data/             # Data processing
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── models/           # Model training and evaluation
│   └── utils/            # Utilities
├── data/                 # Data files
│   └── iris.csv         # Sample dataset
├── tests/               # Test files
│   ├── test_api.py
│   └── test_models.py
├── docker/              # Docker configurations
│   ├── docker-compose.yml
│   └── Dockerfile.api
├── docs/               # Documentation
├── k8s/                # Kubernetes manifests
│   ├── deployment.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   └── namespace.yaml
├── monitoring/         # Monitoring configurations
│   ├── grafana/       # Grafana dashboards
│   └── prometheus/    # Prometheus configuration
├── mlflow/            # MLflow project configuration
├── mlruns/            # MLflow experiment tracking
├── requirements/      # Dependencies
│   ├── api.txt
│   ├── base.txt
│   └── training.txt
├── generate_sample_data.py
├── make.py            # Build and management scripts
└── promote_model.py   # Model promotion utilities
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
