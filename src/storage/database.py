"""SQLite database for job metadata persistence."""

from __future__ import annotations

from datetime import datetime

import aiosqlite
import structlog

from src.storage.models import JobRecord

logger = structlog.get_logger()

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    config TEXT NOT NULL,
    k8s_job_name TEXT,
    retries INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT
)
"""


class JobDatabase:
    """Async SQLite database for training job metadata."""

    def __init__(self, db_path: str = "jobs.db") -> None:
        self.db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        if self._initialized:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(CREATE_TABLE_SQL)
            await db.commit()
        self._initialized = True
        logger.info("database_initialized", path=self.db_path)

    async def create_job(self, record: JobRecord) -> JobRecord:
        """Insert a new job record."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO jobs (id, name, status, config, k8s_job_name, retries,
                   created_at, updated_at, started_at, completed_at, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.name,
                    record.status,
                    record.config,
                    record.k8s_job_name,
                    record.retries,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.started_at.isoformat() if record.started_at else None,
                    record.completed_at.isoformat() if record.completed_at else None,
                    record.error,
                ),
            )
            await db.commit()
        logger.info("job_created", job_id=record.id, name=record.name)
        return record

    async def get_job(self, job_id: str) -> JobRecord | None:
        """Get a job by ID."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    async def list_jobs(self, status_filter: str | None = None) -> list[JobRecord]:
        """List all jobs, optionally filtered by status."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if status_filter:
                cursor = await db.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC",
                    (status_filter,),
                )
            else:
                cursor = await db.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        error: str | None = None,
        k8s_job_name: str | None = None,
        retries: int | None = None,
    ) -> JobRecord | None:
        """Update job status and related fields."""
        await self.initialize()
        now = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            updates = ["status = ?", "updated_at = ?"]
            params: list = [status, now]

            if error is not None:
                updates.append("error = ?")
                params.append(error)

            if k8s_job_name is not None:
                updates.append("k8s_job_name = ?")
                params.append(k8s_job_name)

            if retries is not None:
                updates.append("retries = ?")
                params.append(retries)

            if status == "RUNNING":
                updates.append("started_at = ?")
                params.append(now)
            elif status in ("SUCCEEDED", "FAILED", "CANCELLED", "DEAD_LETTERED"):
                updates.append("completed_at = ?")
                params.append(now)

            params.append(job_id)
            await db.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            await db.commit()

        logger.info("job_status_updated", job_id=job_id, status=status)
        return await self.get_job(job_id)

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job record."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            await db.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info("job_deleted", job_id=job_id)
        return deleted

    async def check_connection(self) -> bool:
        """Check if the database is accessible."""
        try:
            await self.initialize()
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("SELECT 1")
            return True
        except Exception:
            return False

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> JobRecord:
        """Convert a database row to a JobRecord."""

        def parse_dt(val: str | None) -> datetime | None:
            if val is None:
                return None
            return datetime.fromisoformat(val)

        return JobRecord(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            config=row["config"],
            k8s_job_name=row["k8s_job_name"],
            retries=row["retries"],
            created_at=parse_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=parse_dt(row["updated_at"]) or datetime.utcnow(),
            started_at=parse_dt(row["started_at"]),
            completed_at=parse_dt(row["completed_at"]),
            error=row["error"],
        )
