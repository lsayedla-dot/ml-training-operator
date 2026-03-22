"""Pydantic request/response models for the Training Job API."""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class JobStatus(str, enum.Enum):
    """Training job lifecycle states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"
    DEAD_LETTERED = "DEAD_LETTERED"


class ResourceRequirements(BaseModel):
    """K8s resource requests/limits."""

    cpu: str = Field(default="2", examples=["2", "4"])
    memory: str = Field(default="4Gi", examples=["4Gi", "8Gi"])

    @field_validator("cpu")
    @classmethod
    def validate_cpu(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("CPU must not be empty")
        try:
            if v.endswith("m"):
                int(v[:-1])
            else:
                float(v)
        except ValueError:
            raise ValueError(f"Invalid CPU format: {v!r}. Use '2', '500m', etc.")
        return v

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Memory must not be empty")
        valid_suffixes = ("Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "k", "M", "G", "T", "P", "E")
        has_suffix = any(v.endswith(s) for s in valid_suffixes)
        if has_suffix:
            numeric = v
            for s in sorted(valid_suffixes, key=len, reverse=True):
                if numeric.endswith(s):
                    numeric = numeric[: -len(s)]
                    break
            try:
                int(numeric)
            except ValueError:
                raise ValueError(f"Invalid memory format: {v!r}. Use '4Gi', '512Mi', etc.")
        else:
            try:
                int(v)
            except ValueError:
                raise ValueError(f"Invalid memory format: {v!r}. Use '4Gi', '512Mi', etc.")
        return v


class TrainingJobRequest(BaseModel):
    """Request to submit a new training job."""

    name: str = Field(..., min_length=1, max_length=63, examples=["nuscenes-detection-v3"])
    model_type: str = Field(default="resnet18", pattern=r"^(resnet18|resnet50)$")
    dataset: str = Field(default="nuscenes-mini")
    epochs: int = Field(default=10, gt=0, le=1000)
    batch_size: int = Field(default=32, gt=0, le=4096)
    learning_rate: float = Field(default=0.001, gt=0, lt=10)
    num_workers: int = Field(default=1, ge=1, le=64)
    enable_optimization: bool = Field(default=False)
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    checkpoint_interval: int = Field(default=2, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)


class TrainingJobResponse(BaseModel):
    """Response for a training job."""

    id: str
    name: str
    status: JobStatus
    model_type: str
    dataset: str
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    enable_optimization: bool
    resources: ResourceRequirements
    checkpoint_interval: int
    max_retries: int
    retries: int = 0
    k8s_job_name: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    k8s_connected: bool
    db_connected: bool


class ErrorResponse(BaseModel):
    """Structured error response."""

    detail: str
    status_code: int
