"""Tests for SQLite job database operations."""

from __future__ import annotations

import asyncio

import pytest

from src.storage.database import JobDatabase
from src.storage.models import JobRecord


@pytest.fixture
def job_record() -> JobRecord:
    return JobRecord(
        id="db-test-1",
        name="db-test-job",
        status="PENDING",
        config='{"model_type": "resnet18", "epochs": 5}',
    )


@pytest.mark.asyncio
async def test_create_and_get(temp_db, job_record):
    """Create job → retrievable by ID."""
    await temp_db.create_job(job_record)
    result = await temp_db.get_job("db-test-1")
    assert result is not None
    assert result.id == "db-test-1"
    assert result.name == "db-test-job"


@pytest.mark.asyncio
async def test_update_status(temp_db, job_record):
    """Update status → reflected in get."""
    await temp_db.create_job(job_record)
    await temp_db.update_job_status("db-test-1", "RUNNING")
    result = await temp_db.get_job("db-test-1")
    assert result.status == "RUNNING"
    assert result.started_at is not None


@pytest.mark.asyncio
async def test_list_with_filter(temp_db):
    """List with filter returns correct results."""
    await temp_db.create_job(JobRecord(id="f1", name="job1", status="RUNNING", config="{}"))
    await temp_db.create_job(JobRecord(id="f2", name="job2", status="PENDING", config="{}"))
    await temp_db.create_job(JobRecord(id="f3", name="job3", status="RUNNING", config="{}"))

    running = await temp_db.list_jobs(status_filter="RUNNING")
    assert len(running) == 2
    assert all(j.status == "RUNNING" for j in running)


@pytest.mark.asyncio
async def test_delete_job(temp_db, job_record):
    """Delete job → no longer retrievable."""
    await temp_db.create_job(job_record)
    deleted = await temp_db.delete_job("db-test-1")
    assert deleted is True
    result = await temp_db.get_job("db-test-1")
    assert result is None


@pytest.mark.asyncio
async def test_concurrent_operations(temp_db):
    """Concurrent operations don't corrupt the database."""
    jobs = [
        JobRecord(id=f"concurrent-{i}", name=f"job-{i}", status="PENDING", config="{}")
        for i in range(10)
    ]

    # Create all jobs concurrently
    await asyncio.gather(*[temp_db.create_job(j) for j in jobs])

    all_jobs = await temp_db.list_jobs()
    assert len(all_jobs) == 10


@pytest.mark.asyncio
async def test_auto_creates_table(tmp_path):
    """Auto-creates table on first use."""
    db = JobDatabase(db_path=str(tmp_path / "fresh.db"))
    record = JobRecord(id="auto-1", name="auto-test", status="PENDING", config="{}")
    await db.create_job(record)
    result = await db.get_job("auto-1")
    assert result is not None
