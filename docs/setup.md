# Local Setup Guide

## Prerequisites

- Docker Desktop >= 24.0
- Docker Compose >= 2.20
- Python 3.10+
- make (macOS: included; Ubuntu: `sudo apt install make`)
- (Optional for k8s) kubectl + a local cluster (minikube, k3d, or kind)

---

## 1. Clone and Configure

```bash
git clone https://github.com/YOUR_ORG/mlops-pipeline.git
cd mlops-pipeline

# Create your .env from the example template
make init
# or: cp .env.example .env
```

Edit `.env` if you need custom credentials. The defaults work out-of-the-box for local development.

---

## 2. Start the Full Stack

```bash
make up
```

This starts all 9 services:

| Service | URL | Credentials |
|---|---|---|
| Airflow Webserver | http://localhost:8080 | airflow / airflow |
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Model API | http://localhost:8000 | — |

Wait ~2-3 minutes for all services to become healthy. Check status with:

```bash
make ps
# or: docker-compose ps
```

---

## 3. Run the Data Ingestion Pipeline

Trigger the ingestion DAG to generate synthetic training data:

```bash
make ingest
```

Monitor progress at http://localhost:8080. The DAG will:
1. Generate 10,000 synthetic customer records
2. Validate schema and data quality
3. Save to `data/raw/YYYY-MM-DD/churn_raw.csv`
4. Store reference dataset for drift detection

---

## 4. Train the Model

```bash
make train
```

This triggers the `training_pipeline` DAG which runs:
1. **extract** – load latest raw data
2. **transform** – fit preprocessor, generate train/val/test splits
3. **train** – train RandomForest with MLflow autolog
4. **evaluate** – compute test metrics + confusion matrix
5. **register_model** – register to MLflow registry, promote to Staging if AUC > 0.75

Track the experiment at http://localhost:5000.

---

## 5. Promote Model to Production

In the MLflow UI (http://localhost:5000):
1. Go to **Models** → **ChurnModel**
2. Find the Staging version
3. Click **Transition to** → **Production**

Or via Python:
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage("ChurnModel", version="1", stage="Production")
```

---

## 6. Test the Model API

```bash
# Health check
make health

# Single prediction
make predict

# Or use curl directly:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "tenure_months": 6,
    "monthly_charges": 95.0,
    "total_charges": 570.0,
    "num_products": 1,
    "has_tech_support": 0,
    "has_online_security": 0,
    "has_backup": 0,
    "has_device_protection": 0,
    "is_senior_citizen": 0,
    "has_partner": 0,
    "has_dependents": 0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic"
  }'
```

Expected response:
```json
{
  "prediction": 1,
  "probability": 0.823,
  "model_name": "ChurnModel",
  "model_version": "1"
}
```

---

## 7. Run Drift Detection

```bash
make monitor
```

Reports are saved to `monitoring/evidently_reports/YYYY-MM-DD/`.

If drift is detected (share of drifted columns > 0.3), the monitoring service will:
1. Save HTML reports (data_drift.html, data_quality.html, target_drift.html)
2. Trigger the `retraining_pipeline` DAG via Airflow REST API
3. Send a Slack notification (if `SLACK_WEBHOOK_URL` is configured in `.env`)

---

## 8. Run Tests

```bash
make test
```

This runs pytest with 80% coverage requirement. Tests use mocked MLflow and Airflow so no services need to be running.

---

## 9. Stop Everything

```bash
make down        # stop containers, keep volumes
make clean       # stop + delete all volumes (WARNING: deletes all data)
```

---

## Kubernetes Deployment

### Prerequisites
- kubectl configured against a cluster
- Container registry credentials

### Deploy

```bash
# Create namespace and secrets
kubectl apply -f k8s/namespace.yaml
kubectl create secret generic mlops-secrets \
  --from-literal=AWS_ACCESS_KEY_ID=your_key \
  --from-literal=AWS_SECRET_ACCESS_KEY=your_secret \
  -n mlops

# Deploy all manifests
make deploy
```

### Monitor
```bash
kubectl get pods -n mlops
kubectl get hpa -n mlops
kubectl logs -f deployment/model-serving -n mlops
```
