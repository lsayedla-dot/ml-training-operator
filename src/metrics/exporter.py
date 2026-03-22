"""Prometheus metrics for the training job operator."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Job submission metrics
jobs_submitted_total = Counter(
    "training_jobs_submitted_total",
    "Total number of training jobs submitted",
    ["model_type"],
)

# Job completion metrics
jobs_completed_total = Counter(
    "training_jobs_completed_total",
    "Total number of training jobs completed",
    ["model_type", "status"],
)

# Job duration
job_duration_seconds = Histogram(
    "training_job_duration_seconds",
    "Duration of training jobs in seconds",
    ["model_type"],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, float("inf")),
)

# Active jobs gauge
jobs_active = Gauge(
    "training_jobs_active",
    "Number of currently active training jobs",
)

# Retry metrics
job_retries_total = Counter(
    "training_job_retries_total",
    "Total number of job retries",
)
