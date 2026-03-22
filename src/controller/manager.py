"""Job Manager — creates, monitors, and manages training jobs on Kubernetes."""

from __future__ import annotations

import json
import uuid

import structlog

from src.api.models import JobStatus, TrainingJobRequest
from src.controller.job_spec import build_distributed_job_specs, build_single_job_spec
from src.controller.k8s_client import K8sClient
from src.controller.retry import RetryPolicy
from src.metrics.exporter import (
    job_duration_seconds,
    job_retries_total,
    jobs_active,
    jobs_completed_total,
    jobs_submitted_total,
)
from src.storage.database import JobDatabase
from src.storage.models import JobRecord

logger = structlog.get_logger()


class JobManager:
    """Manages the lifecycle of ML training jobs on Kubernetes."""

    def __init__(
        self,
        database: JobDatabase,
        k8s_client: K8sClient | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.database = database
        self.k8s = k8s_client or K8sClient()
        self.retry_policy = retry_policy or RetryPolicy()

    def check_k8s_connection(self) -> bool:
        """Check if the K8s client is connected."""
        return self.k8s.is_connected

    async def create_training_job(self, request: TrainingJobRequest) -> JobRecord:
        """Create a new training job and submit it to Kubernetes."""
        job_id = uuid.uuid4().hex[:12]
        config = request.model_dump()

        record = JobRecord(
            id=job_id,
            name=request.name,
            status=JobStatus.PENDING.value,
            config=json.dumps(config),
        )

        record = await self.database.create_job(record)
        jobs_submitted_total.labels(model_type=request.model_type).inc()
        jobs_active.inc()

        # Submit to K8s if connected
        if self.k8s.is_connected:
            try:
                if request.num_workers > 1:
                    self.k8s.create_headless_service(job_id, request.num_workers)
                    k8s_jobs = build_distributed_job_specs(request, job_id)
                    for k8s_job in k8s_jobs:
                        self.k8s.create_namespaced_job(k8s_job)
                    k8s_job_name = k8s_jobs[0].metadata.name.rsplit("-rank", 1)[0]
                else:
                    k8s_job = build_single_job_spec(request, job_id)
                    result = self.k8s.create_namespaced_job(k8s_job)
                    k8s_job_name = result.metadata.name

                await self.database.update_job_status(
                    job_id,
                    JobStatus.RUNNING.value,
                    k8s_job_name=k8s_job_name,
                )
                record = await self.database.get_job(job_id)
            except Exception as e:
                logger.error("job_submission_failed", job_id=job_id, error=str(e))
                await self.database.update_job_status(job_id, JobStatus.FAILED.value, error=str(e))
                jobs_active.dec()
                record = await self.database.get_job(job_id)

        logger.info("training_job_created", job_id=job_id, name=request.name)
        return record

    async def cancel_job(self, job_id: str) -> JobRecord | None:
        """Cancel a running training job."""
        record = await self.database.get_job(job_id)
        if record is None:
            return None

        # Delete K8s resources
        if self.k8s.is_connected and record.k8s_job_name:
            try:
                config = json.loads(record.config)
                num_workers = config.get("num_workers", 1)

                if num_workers > 1:
                    self.k8s.delete_headless_service(job_id)
                    # Delete all rank jobs
                    for rank in range(num_workers):
                        try:
                            self.k8s.delete_namespaced_job(f"{record.k8s_job_name}-rank{rank}")
                        except Exception:
                            pass
                else:
                    self.k8s.delete_namespaced_job(record.k8s_job_name)
            except Exception as e:
                logger.warning("k8s_cleanup_error", job_id=job_id, error=str(e))

        record = await self.database.update_job_status(job_id, JobStatus.CANCELLED.value)
        jobs_active.dec()
        logger.info("job_cancelled", job_id=job_id)
        return record

    async def get_status(self, job_id: str) -> JobRecord | None:
        """Get the current status of a training job."""
        return await self.database.get_job(job_id)

    async def handle_failure(self, job_id: str, error: str) -> None:
        """Handle a job failure — retry or dead-letter."""
        record = await self.database.get_job(job_id)
        if record is None:
            return

        if self.retry_policy.should_retry(record):
            new_retries = record.retries + 1
            job_retries_total.inc()
            await self.database.update_job_status(
                job_id,
                JobStatus.RETRYING.value,
                retries=new_retries,
                error=error,
            )
            logger.info(
                "job_retrying",
                job_id=job_id,
                retry=new_retries,
                max_retries=self.retry_policy.max_retries,
            )
        else:
            await self.database.update_job_status(
                job_id,
                JobStatus.DEAD_LETTERED.value,
                error=error,
            )
            jobs_active.dec()
            config = json.loads(record.config)
            jobs_completed_total.labels(
                model_type=config.get("model_type", "unknown"),
                status="dead_lettered",
            ).inc()
            logger.warning("job_dead_lettered", job_id=job_id, error=error)

    async def handle_success(self, job_id: str) -> None:
        """Handle a successful job completion."""
        record = await self.database.get_job(job_id)
        if record is None:
            return

        await self.database.update_job_status(job_id, JobStatus.SUCCEEDED.value)
        jobs_active.dec()

        config = json.loads(record.config)
        jobs_completed_total.labels(
            model_type=config.get("model_type", "unknown"),
            status="succeeded",
        ).inc()

        if record.started_at:
            from datetime import datetime

            duration = (datetime.utcnow() - record.started_at).total_seconds()
            job_duration_seconds.labels(model_type=config.get("model_type", "unknown")).observe(
                duration
            )

        logger.info("job_succeeded", job_id=job_id)

    async def cleanup_completed(self, ttl_hours: int = 24) -> int:
        """Clean up K8s resources for jobs completed more than ttl_hours ago."""
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(hours=ttl_hours)
        cleaned = 0

        terminal_statuses = [
            JobStatus.SUCCEEDED.value,
            JobStatus.FAILED.value,
            JobStatus.DEAD_LETTERED.value,
        ]
        for status in terminal_statuses:
            records = await self.database.list_jobs(status_filter=status)
            for record in records:
                if record.completed_at and record.completed_at < cutoff:
                    if self.k8s.is_connected and record.k8s_job_name:
                        try:
                            config = json.loads(record.config)
                            num_workers = config.get("num_workers", 1)
                            if num_workers > 1:
                                self.k8s.delete_headless_service(record.id)
                                for rank in range(num_workers):
                                    try:
                                        self.k8s.delete_namespaced_job(
                                            f"{record.k8s_job_name}-rank{rank}"
                                        )
                                    except Exception:
                                        pass
                            else:
                                self.k8s.delete_namespaced_job(record.k8s_job_name)
                        except Exception as e:
                            logger.warning("cleanup_error", job_id=record.id, error=str(e))
                    cleaned += 1

        logger.info("cleanup_completed", jobs_cleaned=cleaned)
        return cleaned
