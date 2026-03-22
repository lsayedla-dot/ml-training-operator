"""Tests for Pydantic request/response model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models import JobStatus, ResourceRequirements, TrainingJobRequest, TrainingJobResponse


def test_valid_request():
    """Valid request passes validation."""
    req = TrainingJobRequest(
        name="test-job",
        model_type="resnet18",
        epochs=10,
        batch_size=32,
    )
    assert req.name == "test-job"
    assert req.num_workers == 1


def test_negative_epochs_rejected():
    """Negative epochs are rejected."""
    with pytest.raises(ValidationError):
        TrainingJobRequest(name="test", epochs=-1)


def test_empty_name_rejected():
    """Empty name is rejected."""
    with pytest.raises(ValidationError):
        TrainingJobRequest(name="", epochs=5)


def test_num_workers_must_be_positive():
    """num_workers must be >= 1."""
    with pytest.raises(ValidationError):
        TrainingJobRequest(name="test", num_workers=0)


def test_resource_format_validated():
    """Resource format must be valid K8s format."""
    # Valid formats
    ResourceRequirements(cpu="2", memory="4Gi")
    ResourceRequirements(cpu="500m", memory="512Mi")

    # Invalid formats
    with pytest.raises(ValidationError):
        ResourceRequirements(cpu="abc", memory="4Gi")
    with pytest.raises(ValidationError):
        ResourceRequirements(cpu="2", memory="bad")


def test_response_serializes():
    """Response model serializes correctly."""
    from datetime import datetime

    resp = TrainingJobResponse(
        id="abc123",
        name="test",
        status=JobStatus.RUNNING,
        model_type="resnet18",
        dataset="nuscenes-mini",
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        num_workers=1,
        enable_optimization=False,
        resources=ResourceRequirements(),
        checkpoint_interval=2,
        max_retries=3,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    data = resp.model_dump()
    assert data["id"] == "abc123"
    assert data["status"] == "RUNNING"
