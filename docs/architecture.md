# Architecture Overview

## System Architecture Diagram

```
                          ┌─────────────────────────────────────────────────┐
                          │              Docker Network: mlops-network        │
                          │                                                   │
  External Traffic        │   ┌────────────┐     ┌──────────────────────┐   │
  ─────────────────────►  │   │  FastAPI   │     │   Apache Airflow     │   │
        Port 8000         │   │  Serving   │     │   (Webserver+        │   │
                          │   │  :8000     │     │    Scheduler+Worker) │   │
                          │   └─────┬──────┘     └──────────┬───────────┘   │
                          │         │                        │               │
                          │         │ load model             │ orchestrate   │
                          │         ▼                        │               │
                          │   ┌─────────────┐               │               │
                          │   │   MLflow    │◄──────────────┘               │
                          │   │   Server   │  log runs/                     │
                          │   │   :5000    │  register models               │
                          │   └──────┬─────┘                                │
                          │          │ artifacts                            │
                          │          ▼          metrics/metadata            │
                          │   ┌─────────────┐   ┌──────────────┐           │
                          │   │   MinIO     │   │  PostgreSQL  │           │
                          │   │   (S3)      │   │  (metadata + │           │
                          │   │   :9000     │   │   registry)  │           │
                          │   └─────────────┘   └──────────────┘           │
                          │                                                   │
                          │   ┌─────────────┐   ┌──────────────┐           │
                          │   │  Monitoring │   │    Redis     │           │
                          │   │  (Evidently)│   │  (Celery     │           │
                          │   │   + Cron    │   │   broker)    │           │
                          │   └─────────────┘   └──────────────┘           │
                          └─────────────────────────────────────────────────┘
```

## Data Flow

### Training Pipeline
```
Raw Data (synthetic/real)
        │
        ▼
┌──────────────┐    date-partitioned CSV
│   Ingestion  │ ──────────────────────► data/raw/YYYY-MM-DD/
│     DAG      │
└──────┬───────┘
       │ validate schema
       ▼
┌──────────────┐    processed arrays
│   Training   │ ──────────────────────► data/processed/
│     DAG      │         + preprocessor.pkl
│              │
│  1. extract  │
│  2. transform│
│  3. train    │ ──────────────────────► MLflow Experiment Runs
│  4. evaluate │                          (params, metrics, artifacts)
│  5. register │ ──────────────────────► MLflow Model Registry
└──────────────┘                          (ChurnModel → Staging/Production)
```

### Serving Pipeline
```
HTTP POST /predict
        │
        ▼
┌──────────────┐
│  FastAPI     │
│  app.py      │
│              │
│ 1. Validate  │ (Pydantic schema)
│ 2. Transform │ (pipeline includes preprocessor)
│ 3. Predict   │ ──────────────────────► prediction + probability
│ 4. Log       │ ──────────────────────► data/predictions.csv
└──────────────┘
```

### Monitoring + Retraining Pipeline
```
Scheduled (daily at 1am)
        │
        ▼
┌──────────────┐
│  Drift       │ ◄── data/predictions.csv (last 24h)
│  Detection   │ ◄── data/processed/reference.csv (training data)
│              │
│  Evidently   │ ──────────────────────► monitoring/evidently_reports/
│  Reports:    │                          data_drift.html
│  - Data Drift│                          data_quality.html
│  - Quality   │                          target_drift.html
└──────┬───────┘
       │ drift_score > 0.3?
       │
   YES │                    NO
       ▼                    ▼
┌──────────────┐      No action
│  Alerting    │
│              │
│ 1. Slack     │ ──────────────────────► #mlops-alerts channel
│ 2. Trigger   │ ──────────────────────► Airflow REST API
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│         Retraining DAG                    │
│                                           │
│ 1. check_trigger (drift/scheduled)        │
│ 2. load_combined_data                     │
│    (historical + labeled production data) │
│ 3. retrain (same pipeline + drift tag)    │
│ 4. promote_model (champion/challenger)    │
│ 5. notify_completion (Slack)              │
└──────────────────────────────────────────┘
```

## Component Details

### Apache Airflow
- **Executor**: CeleryExecutor with Redis broker for distributed task execution
- **DAGs**:
  - `data_ingestion` (daily): Generate/validate/store raw data
  - `training_pipeline` (weekly): Full train-evaluate-register cycle
  - `retraining_pipeline` (weekly + on-demand): Drift-triggered retraining
- **REST API**: Used by the monitoring service to programmatically trigger DAG runs

### MLflow
- **Tracking Server**: Central experiment tracking with PostgreSQL backend
- **Artifact Store**: MinIO (S3-compatible) for models, plots, and preprocessors
- **Model Registry**: Version management with Staging → Production transitions
- **Autolog**: scikit-learn autolog captures all parameters and metrics automatically

### FastAPI Model Server
- **Startup**: Loads `Production` model from MLflow registry via `lifespan` hook
- **Prediction Pipeline**: The loaded sklearn Pipeline object includes the preprocessor — raw features go in, predictions come out
- **Prediction Store**: Every request is appended to `predictions.csv` for drift monitoring
- **Prometheus Metrics**: Custom `/metrics` endpoint for observability

### Evidently AI
- **DataDriftReport**: Detects statistical shifts in feature distributions
- **DataQualityReport**: Checks for nulls, outliers, and schema violations
- **TargetDriftReport**: Monitors prediction distribution shifts
- **Threshold**: `DRIFT_THRESHOLD=0.3` (30% of features drifting triggers retraining)

### Kubernetes
- **Deployment**: 2 replicas with rolling update strategy
- **HPA**: Scales 2-10 pods based on CPU (70%) and memory (80%) utilization
- **Health probes**: Liveness + readiness on `/health` with proper initial delays
- **Resource limits**: 500m CPU / 512Mi memory per pod

### CI/CD (GitHub Actions)
- **CI** (on PR): Lint → Test (80% coverage) → Build images (no push)
- **CD** (on main merge): Build → Push to GHCR → `kubectl set image` → Apply manifests → Smoke test

## Key Design Decisions

1. **Pipeline includes preprocessor**: The sklearn `Pipeline` wraps both the `ColumnTransformer` and the classifier. This means the model artifact in MLflow is self-contained — no need to separately load and apply the preprocessor at serving time.

2. **Date-partitioned raw data**: Raw data is stored as `data/raw/YYYY-MM-DD/churn_raw.csv`. This enables easy reprocessing of historical data, auditability, and time-travel for debugging.

3. **Champion/challenger promotion**: New models only replace Production if AUC improves by >1%. This prevents regression from noisy training runs.

4. **Predictions as ground truth store**: Every prediction is logged to CSV with a UUID. The `/feedback` endpoint allows writing back actual labels. The monitoring service uses labeled predictions to detect target drift.

5. **Environment variable configuration**: All service URLs, credentials, and thresholds are environment variables with sensible defaults pointing to docker-compose services. The same code runs locally (docker-compose) and in Kubernetes (ConfigMap + Secrets).
