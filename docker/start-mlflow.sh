#!/bin/bash
set -e

pip install mlflow boto3 psycopg2-binary --quiet

mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port 5000
