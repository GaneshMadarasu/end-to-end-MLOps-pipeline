# End-to-End MLOps Pipeline

A production-grade MLOps pipeline for customer churn prediction, demonstrating the complete lifecycle from raw data to deployed model with automated drift detection and retraining.

## Architecture

```
Data Ingestion → Feature Engineering → Training → Evaluation → Serving
     (Airflow)       (sklearn)          (MLflow)   (MLflow)   (FastAPI)
                                                               ↓
                                                    Monitoring (Evidently)
                                                               ↓
                                               Drift Alert → Retraining
```

## Stack

| Component | Technology |
|---|---|
| Orchestration | Apache Airflow 2.8 (Celery + Redis) |
| Experiment Tracking | MLflow 2.12 + PostgreSQL + MinIO |
| Model Serving | FastAPI + Uvicorn |
| Containerization | Docker Compose (local) + Kubernetes (prod) |
| Monitoring | Evidently AI |
| CI/CD | GitHub Actions → GHCR → kubectl |
| Language | Python 3.10+ |

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/YOUR_ORG/mlops-pipeline.git
cd mlops-pipeline
cp .env.example .env

# 2. Start all services
make up

# 3. Generate data and train the model
make ingest
make train

# 4. Promote the model to Production in MLflow UI (http://localhost:5000)

# 5. Make predictions
make predict

# 6. Run drift detection
make monitor
```

## Service URLs

| Service | URL | Default Credentials |
|---|---|---|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Model API | http://localhost:8000 | — |
| API Docs | http://localhost:8000/docs | — |

## Project Structure

```
mlops-pipeline/
├── dags/                    # Airflow DAGs
│   ├── ingestion_dag.py     # Daily data ingestion & validation
│   ├── training_dag.py      # Weekly train → evaluate → register
│   └── retraining_dag.py    # Drift-triggered or scheduled retraining
├── src/
│   ├── data/                # Ingestion + validation
│   ├── features/            # Feature engineering pipeline
│   ├── training/            # Training + hyperparameter optimization
│   ├── evaluation/          # Metrics + champion/challenger promotion
│   ├── serving/             # FastAPI app + model loader + schemas
│   └── monitoring/          # Evidently drift detection + alerting
├── k8s/                     # Kubernetes manifests
├── docker/                  # Dockerfiles + init scripts
├── tests/                   # Pytest test suite (>80% coverage)
├── .github/workflows/       # CI (lint+test) + CD (build+deploy)
├── docs/                    # Setup guide + architecture docs
├── docker-compose.yml       # Full local stack
├── Makefile                 # Developer shortcuts
└── requirements.txt         # Pinned Python dependencies
```

## Pipeline Overview

### 1. Data Ingestion (`dags/ingestion_dag.py`)
Generates 10,000 synthetic customer churn records with 15 features (12 numerical + 3 categorical), validates schema and data quality, and stores partitioned by date under `data/raw/YYYY-MM-DD/`.

### 2. Feature Engineering (`src/features/engineering.py`)
Fits a `ColumnTransformer` (StandardScaler + OneHotEncoder) on the training split and wraps it with the classifier in an sklearn `Pipeline`. The pipeline artifact is logged to MLflow so serving uses identical transforms.

### 3. Training (`dags/training_dag.py`)
Five tasks: **extract → transform → train → evaluate → register_model**. Uses MLflow autolog for sklearn, logs confusion matrix and ROC curve as artifacts, and auto-promotes to Staging if validation AUC > 0.75.

### 4. Promotion (`src/evaluation/promotion.py`)
Champion/challenger comparison on a holdout set. The challenger replaces Production only if AUC improves by >1%. Old Production models are transitioned to Archived.

### 5. Serving (`src/serving/app.py`)
FastAPI application with four endpoints:
- `GET /health` — model metadata and status
- `POST /predict` — churn prediction with probability
- `GET /metrics` — Prometheus-format metrics
- `POST /feedback` — record ground truth labels

### 6. Monitoring (`src/monitoring/drift_detector.py`)
Daily Evidently AI reports comparing production predictions against reference data:
- DataDriftReport (feature distribution shift)
- DataQualityReport (nulls, outliers, schema)
- TargetDriftReport (prediction distribution)

If >30% of features drift, the retraining pipeline is triggered automatically.

### 7. Retraining (`dags/retraining_dag.py`)
Combines historical training data with labeled production predictions, retrains the model, runs champion/challenger promotion, and sends Slack notification on completion.

## API Reference

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "tenure_months": 24,
    "monthly_charges": 65.5,
    "total_charges": 1572.0,
    "num_products": 2,
    "has_tech_support": 1,
    "has_online_security": 0,
    "has_backup": 1,
    "has_device_protection": 0,
    "is_senior_citizen": 0,
    "has_partner": 1,
    "has_dependents": 0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic"
  }'
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.823456,
  "model_name": "ChurnModel",
  "model_version": "1"
}
```

## Make Targets

```
make up          Start all services
make down        Stop all services
make build       Build Docker images
make ingest      Trigger ingestion DAG
make train       Trigger training DAG
make monitor     Run drift detection manually
make retrain     Trigger retraining DAG
make deploy      kubectl apply -f k8s/
make test        Run pytest with coverage
make lint        Lint with ruff
make predict     Send sample prediction request
make health      Check API health
make logs        Tail model-serving logs
make clean       Remove all containers and volumes
```

## Development

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format

# View all logs
make logs-all
```

## Kubernetes Deployment

```bash
# Create namespace and secrets
kubectl apply -f k8s/namespace.yaml
kubectl create secret generic mlops-secrets \
  --from-literal=AWS_ACCESS_KEY_ID=YOUR_KEY \
  --from-literal=AWS_SECRET_ACCESS_KEY=YOUR_SECRET \
  -n mlops

# Deploy
make deploy

# Check status
kubectl get pods -n mlops
kubectl get hpa -n mlops
```

See [docs/setup.md](docs/setup.md) for complete setup instructions and [docs/architecture.md](docs/architecture.md) for architecture details.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MODEL_NAME` | `ChurnModel` | Registered model name |
| `MODEL_STAGE` | `Production` | Model stage to load |
| `DRIFT_THRESHOLD` | `0.3` | Fraction of drifted features to trigger retraining |
| `SLACK_WEBHOOK_URL` | — | Slack webhook for alerts |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO access key |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO secret key |

## License

MIT
