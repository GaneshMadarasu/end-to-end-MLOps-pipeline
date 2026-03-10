.PHONY: up down train monitor deploy test logs clean build help

AIRFLOW_API=http://localhost:8080/api/v1
AIRFLOW_USER=airflow
AIRFLOW_PASS=airflow

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up:  ## Start all services (docker-compose up -d)
	docker-compose up -d
	@echo "Services starting..."
	@echo "  Airflow:  http://localhost:8080 (airflow/airflow)"
	@echo "  MLflow:   http://localhost:5000"
	@echo "  MinIO:    http://localhost:9001 (minioadmin/minioadmin)"
	@echo "  API:      http://localhost:8000"

down:  ## Stop all services
	docker-compose down

build:  ## Build all custom Docker images
	docker-compose build

train:  ## Trigger training DAG via Airflow CLI
	curl -s -X POST \
		"$(AIRFLOW_API)/dags/training_pipeline/dagRuns" \
		-H "Content-Type: application/json" \
		-u "$(AIRFLOW_USER):$(AIRFLOW_PASS)" \
		-d '{"conf": {}}' | python3 -m json.tool
	@echo "Training DAG triggered. Monitor at http://localhost:8080"

ingest:  ## Trigger ingestion DAG via Airflow CLI
	curl -s -X POST \
		"$(AIRFLOW_API)/dags/data_ingestion/dagRuns" \
		-H "Content-Type: application/json" \
		-u "$(AIRFLOW_USER):$(AIRFLOW_PASS)" \
		-d '{"conf": {}}' | python3 -m json.tool

monitor:  ## Run drift detection manually
	docker-compose exec monitoring python -m src.monitoring.drift_detector

retrain:  ## Trigger retraining DAG
	curl -s -X POST \
		"$(AIRFLOW_API)/dags/retraining_pipeline/dagRuns" \
		-H "Content-Type: application/json" \
		-u "$(AIRFLOW_USER):$(AIRFLOW_PASS)" \
		-d '{"conf": {"trigger": "manual"}}' | python3 -m json.tool

deploy:  ## Apply Kubernetes manifests
	kubectl apply -f k8s/

test:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80

lint:  ## Lint with ruff
	ruff check src/ dags/ tests/

format:  ## Format with ruff
	ruff format src/ dags/ tests/

logs:  ## Tail model-serving logs
	docker-compose logs -f model-serving

logs-all:  ## Tail all service logs
	docker-compose logs -f

predict:  ## Send a sample prediction request
	curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"age": 35, "tenure_months": 24, "monthly_charges": 65.5, "total_charges": 1572.0, "num_products": 2, "has_tech_support": 1, "has_online_security": 0, "has_backup": 1, "has_device_protection": 0, "is_senior_citizen": 0, "has_partner": 1, "has_dependents": 0, "contract_type": "Month-to-month", "payment_method": "Electronic check", "internet_service": "Fiber optic"}' \
		| python3 -m json.tool

health:  ## Check model serving health
	curl -s http://localhost:8000/health | python3 -m json.tool

clean:  ## Tear down all containers and volumes
	docker-compose down -v --remove-orphans
	@echo "All containers and volumes removed."

clean-data:  ## Remove generated data files
	rm -rf data/raw/* data/processed/*
	@echo "Data files cleaned."

init:  ## Initialize project (copy .env.example to .env)
	@if [ ! -f .env ]; then cp .env.example .env; echo ".env created from .env.example"; else echo ".env already exists"; fi

ps:  ## Show running services
	docker-compose ps
