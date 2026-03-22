"""Dependency injection for the FastAPI application."""

from __future__ import annotations

from functools import lru_cache

from src.controller.manager import JobManager
from src.storage.database import JobDatabase


@lru_cache
def get_database() -> JobDatabase:
    """Get the singleton database instance."""
    return JobDatabase()


@lru_cache
def get_job_manager() -> JobManager:
    """Get the singleton job manager instance."""
    return JobManager(database=get_database())
