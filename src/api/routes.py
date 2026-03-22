"""API routes for the Training Job API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_database, get_job_manager
from src.api.models import (
    HealthResponse,
    JobStatus,
    TrainingJobRequest,
    TrainingJobResponse,
)
from src.controller.manager import JobManager
from src.storage.database import JobDatabase

router = APIRouter()


def _record_to_response(record, config: dict) -> TrainingJobResponse:
    """Convert a JobRecord + parsed config to a TrainingJobResponse."""
    from src.api.models import ResourceRequirements

    return TrainingJobResponse(
        id=record.id,
        name=record.name,
        status=JobStatus(record.status),
        model_type=config.get("model_type", "resnet18"),
        dataset=config.get("dataset", "nuscenes-mini"),
        epochs=config.get("epochs", 10),
        batch_size=config.get("batch_size", 32),
        learning_rate=config.get("learning_rate", 0.001),
        num_workers=config.get("num_workers", 1),
        enable_optimization=config.get("enable_optimization", False),
        resources=ResourceRequirements(**config.get("resources", {})),
        checkpoint_interval=config.get("checkpoint_interval", 2),
        max_retries=config.get("max_retries", 3),
        retries=record.retries,
        k8s_job_name=record.k8s_job_name,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error=record.error,
    )


@router.post("/jobs", status_code=201, response_model=TrainingJobResponse)
async def submit_job(
    request: TrainingJobRequest,
    manager: JobManager = Depends(get_job_manager),
) -> TrainingJobResponse:
    """Submit a new training job."""
    record = await manager.create_training_job(request)
    import json

    config = json.loads(record.config)
    return _record_to_response(record, config)


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_jobs(
    status: str | None = Query(default=None, description="Filter by job status"),
    db: JobDatabase = Depends(get_database),
) -> list[TrainingJobResponse]:
    """List all training jobs with optional status filter."""
    import json

    records = await db.list_jobs(status_filter=status)
    results = []
    for record in records:
        config = json.loads(record.config)
        results.append(_record_to_response(record, config))
    return results


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_job(
    job_id: str,
    db: JobDatabase = Depends(get_database),
) -> TrainingJobResponse:
    """Get details for a specific training job."""
    import json

    record = await db.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    config = json.loads(record.config)
    return _record_to_response(record, config)


@router.delete("/jobs/{job_id}", response_model=TrainingJobResponse)
async def cancel_job(
    job_id: str,
    manager: JobManager = Depends(get_job_manager),
) -> TrainingJobResponse:
    """Cancel a running training job."""
    import json

    record = await manager.cancel_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    config = json.loads(record.config)
    return _record_to_response(record, config)


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: JobDatabase = Depends(get_database),
    manager: JobManager = Depends(get_job_manager),
) -> HealthResponse:
    """Health check endpoint."""
    db_ok = await db.check_connection()
    k8s_ok = manager.check_k8s_connection()
    status = "healthy" if (db_ok and k8s_ok) else "degraded"
    return HealthResponse(status=status, k8s_connected=k8s_ok, db_connected=db_ok)
