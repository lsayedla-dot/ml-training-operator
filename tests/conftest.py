"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.models import ResourceRequirements, TrainingJobRequest
from src.storage.database import JobDatabase


@pytest.fixture
def sample_job_request() -> TrainingJobRequest:
    """A valid training job request."""
    return TrainingJobRequest(
        name="test-nuscenes-v1",
        model_type="resnet18",
        dataset="nuscenes-mini",
        epochs=5,
        batch_size=16,
        learning_rate=0.001,
        num_workers=1,
        enable_optimization=False,
        resources=ResourceRequirements(cpu="2", memory="4Gi"),
        checkpoint_interval=2,
        max_retries=3,
    )


@pytest.fixture
def distributed_job_request() -> TrainingJobRequest:
    """A distributed training job request."""
    return TrainingJobRequest(
        name="test-distributed",
        model_type="resnet18",
        epochs=5,
        batch_size=16,
        num_workers=4,
        enable_optimization=True,
    )


@pytest.fixture
def invalid_job_requests() -> list[dict]:
    """List of invalid job configurations."""
    return [
        {"name": "test", "epochs": -1},  # negative epochs
        {"name": "", "epochs": 5},  # empty name
        {"name": "test", "num_workers": 0},  # num_workers < 1
        {"name": "test", "resources": {"cpu": "2", "memory": "bad"}},  # bad memory format
        {"name": "test", "model_type": "invalid"},  # invalid model type
    ]


@pytest.fixture
def temp_db(tmp_path) -> JobDatabase:
    """SQLite database in a temp directory."""
    db_path = str(tmp_path / "test_jobs.db")
    return JobDatabase(db_path=db_path)


@pytest.fixture
def mock_k8s_client():
    """Patches kubernetes.client and returns canned responses."""
    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.connect.return_value = True
    mock_job_result = MagicMock()
    mock_job_result.metadata = MagicMock()
    mock_job_result.metadata.name = "train-test-abc12345"
    mock_client.create_namespaced_job.return_value = mock_job_result
    mock_client.read_namespaced_job_status.return_value = MagicMock(
        status=MagicMock(
            succeeded=1,
            failed=None,
            active=None,
        )
    )
    mock_client.delete_namespaced_job.return_value = None
    mock_client.create_headless_service.return_value = MagicMock()
    mock_client.delete_headless_service.return_value = None
    return mock_client


@pytest.fixture
def api_client(temp_db, mock_k8s_client):
    """FastAPI TestClient with mocked dependencies."""
    from fastapi.testclient import TestClient

    from src.api.app import create_app
    from src.api.dependencies import get_database, get_job_manager
    from src.controller.manager import JobManager

    app = create_app()
    manager = JobManager(database=temp_db, k8s_client=mock_k8s_client)

    app.dependency_overrides[get_database] = lambda: temp_db
    app.dependency_overrides[get_job_manager] = lambda: manager

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
