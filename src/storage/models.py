"""Database model definitions for job storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class JobRecord:
    """Database record for a training job."""

    id: str
    name: str
    status: str
    config: str  # JSON-serialized TrainingJobRequest
    k8s_job_name: str | None = None
    retries: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
