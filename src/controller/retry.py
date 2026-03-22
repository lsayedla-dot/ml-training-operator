"""Retry logic with exponential backoff for failed training jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.storage.models import JobRecord


@dataclass
class RetryPolicy:
    """Configuration for job retry behavior."""

    max_retries: int = 3
    base_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0

    def should_retry(self, job: JobRecord) -> bool:
        """Check if the job should be retried based on retry count."""
        return job.retries < self.max_retries

    def next_retry_at(self, job: JobRecord) -> datetime:
        """Calculate the next retry time with exponential backoff."""
        delay = self.base_delay_seconds * (self.backoff_multiplier**job.retries)
        return datetime.utcnow() + timedelta(seconds=delay)

    def get_delay_seconds(self, retry_count: int) -> float:
        """Get the delay in seconds for a given retry attempt."""
        return self.base_delay_seconds * (self.backoff_multiplier**retry_count)
