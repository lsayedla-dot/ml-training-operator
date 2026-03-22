"""Tests for retry logic with exponential backoff."""

from __future__ import annotations

from src.controller.retry import RetryPolicy
from src.storage.models import JobRecord


def _make_job(retries: int = 0) -> JobRecord:
    """Create a test job record."""
    return JobRecord(
        id="test-retry",
        name="retry-test",
        status="FAILED",
        config="{}",
        retries=retries,
    )


def test_first_failure_should_retry():
    """First failure should trigger retry."""
    policy = RetryPolicy(max_retries=3)
    job = _make_job(retries=0)
    assert policy.should_retry(job) is True


def test_second_failure_backoff():
    """Second failure retries after base_delay * multiplier."""
    policy = RetryPolicy(base_delay_seconds=60, backoff_multiplier=2.0)
    delay = policy.get_delay_seconds(1)
    assert delay == 120.0  # 60 * 2^1


def test_max_retries_exceeded():
    """After max retries, should_retry returns False."""
    policy = RetryPolicy(max_retries=3)
    job = _make_job(retries=3)
    assert policy.should_retry(job) is False


def test_backoff_calculation():
    """Backoff calculation is correct for multiple retries."""
    policy = RetryPolicy(base_delay_seconds=60, backoff_multiplier=2.0)
    assert policy.get_delay_seconds(0) == 60.0  # 60 * 2^0
    assert policy.get_delay_seconds(1) == 120.0  # 60 * 2^1
    assert policy.get_delay_seconds(2) == 240.0  # 60 * 2^2


def test_next_retry_at():
    """next_retry_at returns a future datetime."""
    from datetime import datetime

    policy = RetryPolicy(base_delay_seconds=60)
    job = _make_job(retries=0)
    retry_at = policy.next_retry_at(job)
    assert retry_at > datetime.utcnow()
