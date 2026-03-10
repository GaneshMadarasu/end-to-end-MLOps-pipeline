"""Pydantic schemas for FastAPI request and response models."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Input schema for /predict endpoint."""

    age: int = Field(..., ge=18, le=120, description="Customer age in years")
    tenure_months: int = Field(..., ge=0, le=600, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill amount")
    total_charges: float = Field(..., ge=0, description="Total amount charged")
    num_products: int = Field(..., ge=1, le=20, description="Number of products subscribed")
    has_tech_support: int = Field(..., ge=0, le=1, description="Tech support flag (0/1)")
    has_online_security: int = Field(..., ge=0, le=1)
    has_backup: int = Field(..., ge=0, le=1)
    has_device_protection: int = Field(..., ge=0, le=1)
    is_senior_citizen: int = Field(..., ge=0, le=1)
    has_partner: int = Field(..., ge=0, le=1)
    has_dependents: int = Field(..., ge=0, le=1)
    contract_type: str = Field(..., description="Month-to-month | One year | Two year")
    payment_method: str = Field(
        ...,
        description="Electronic check | Mailed check | Bank transfer (automatic) | Credit card (automatic)",
    )
    internet_service: str = Field(..., description="DSL | Fiber optic | No")

    model_config = {"json_schema_extra": {
        "example": {
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
            "internet_service": "Fiber optic",
        }
    }}


class PredictResponse(BaseModel):
    """Output schema for /predict endpoint."""

    prediction: int = Field(..., description="Binary churn prediction (0=No, 1=Yes)")
    probability: float = Field(..., description="Probability of churn")
    model_name: str = Field(..., description="Model registry name")
    model_version: str = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    """Output schema for /health endpoint."""

    status: str
    model_name: str
    model_version: str
    model_stage: str


class FeedbackRequest(BaseModel):
    """Input schema for /feedback endpoint."""

    prediction_id: str = Field(..., description="UUID of the original prediction")
    actual_label: int = Field(..., ge=0, le=1, description="Ground truth label (0 or 1)")


class FeedbackResponse(BaseModel):
    """Output schema for /feedback endpoint."""

    status: str
    prediction_id: str
    message: str


class MetricsResponse(BaseModel):
    """Output schema for /metrics endpoint (Prometheus text format returned as plain text)."""

    pass
