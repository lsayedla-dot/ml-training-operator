"""Build Kubernetes Job specifications from training job requests."""

from __future__ import annotations

import uuid

from kubernetes import client

from src.api.models import TrainingJobRequest

WORKER_IMAGE = "ml-training-worker:latest"
CHECKPOINT_PVC_NAME = "ml-training-checkpoints"
DATA_PVC_NAME = "ml-training-data"
DDP_PORT = 29500


def _short_uuid() -> str:
    """Generate a short UUID for unique job naming."""
    return uuid.uuid4().hex[:8]


def _build_env_vars(
    request: TrainingJobRequest,
    rank: int = 0,
    world_size: int = 1,
    master_addr: str | None = None,
) -> list[client.V1EnvVar]:
    """Build environment variables for a training worker."""
    envs = [
        client.V1EnvVar(name="MODEL_TYPE", value=request.model_type),
        client.V1EnvVar(name="DATASET", value=request.dataset),
        client.V1EnvVar(name="EPOCHS", value=str(request.epochs)),
        client.V1EnvVar(name="BATCH_SIZE", value=str(request.batch_size)),
        client.V1EnvVar(name="LEARNING_RATE", value=str(request.learning_rate)),
        client.V1EnvVar(
            name="CHECKPOINT_INTERVAL",
            value=str(request.checkpoint_interval),
        ),
        client.V1EnvVar(name="CHECKPOINT_DIR", value="/checkpoints"),
        client.V1EnvVar(name="DATA_DIR", value="/data"),
        client.V1EnvVar(
            name="ENABLE_OPTIMIZATION",
            value=str(request.enable_optimization).lower(),
        ),
        client.V1EnvVar(name="WORLD_SIZE", value=str(world_size)),
        client.V1EnvVar(name="RANK", value=str(rank)),
    ]

    if master_addr:
        envs.extend(
            [
                client.V1EnvVar(name="MASTER_ADDR", value=master_addr),
                client.V1EnvVar(name="MASTER_PORT", value=str(DDP_PORT)),
            ]
        )

    return envs


def _build_volume_mounts() -> list[client.V1VolumeMount]:
    """Build volume mounts for checkpoint and data PVCs."""
    return [
        client.V1VolumeMount(name="checkpoints", mount_path="/checkpoints"),
        client.V1VolumeMount(name="data", mount_path="/data"),
    ]


def _build_volumes() -> list[client.V1Volume]:
    """Build volume definitions for PVCs."""
    return [
        client.V1Volume(
            name="checkpoints",
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=CHECKPOINT_PVC_NAME
            ),
        ),
        client.V1Volume(
            name="data",
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=DATA_PVC_NAME
            ),
        ),
    ]


def _build_container(
    request: TrainingJobRequest,
    env_vars: list[client.V1EnvVar],
) -> client.V1Container:
    """Build the training worker container spec."""
    return client.V1Container(
        name="training-worker",
        image=WORKER_IMAGE,
        command=["python", "-m", "src.worker.train"],
        env=env_vars,
        resources=client.V1ResourceRequirements(
            requests={"cpu": request.resources.cpu, "memory": request.resources.memory},
            limits={"cpu": request.resources.cpu, "memory": request.resources.memory},
        ),
        volume_mounts=_build_volume_mounts(),
    )


def build_single_job_spec(request: TrainingJobRequest, job_id: str) -> client.V1Job:
    """Build a K8s Job spec for a single-worker training job."""
    short_id = _short_uuid()
    job_name = f"train-{request.name}-{short_id}"
    env_vars = _build_env_vars(request)

    return client.V1Job(
        metadata=client.V1ObjectMeta(
            name=job_name,
            labels={
                "app": "ml-training-worker",
                "job-id": job_id,
                "model-type": request.model_type,
            },
        ),
        spec=client.V1JobSpec(
            backoff_limit=0,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        "app": "ml-training-worker",
                        "job-id": job_id,
                    },
                ),
                spec=client.V1PodSpec(
                    containers=[_build_container(request, env_vars)],
                    volumes=_build_volumes(),
                    restart_policy="Never",
                    service_account_name="ml-training-operator",
                ),
            ),
        ),
    )


def build_distributed_job_specs(request: TrainingJobRequest, job_id: str) -> list[client.V1Job]:
    """Build K8s Job specs for a distributed DDP training job."""
    short_id = _short_uuid()
    master_addr = f"train-{job_id}-headless.ml-training.svc.cluster.local"
    jobs = []

    for rank in range(request.num_workers):
        job_name = f"train-{request.name}-{short_id}-rank{rank}"
        env_vars = _build_env_vars(
            request,
            rank=rank,
            world_size=request.num_workers,
            master_addr=master_addr,
        )

        job = client.V1Job(
            metadata=client.V1ObjectMeta(
                name=job_name,
                labels={
                    "app": "ml-training-worker",
                    "job-id": job_id,
                    "model-type": request.model_type,
                    "rank": str(rank),
                },
            ),
            spec=client.V1JobSpec(
                backoff_limit=0,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": "ml-training-worker",
                            "job-id": job_id,
                            "rank": str(rank),
                        },
                    ),
                    spec=client.V1PodSpec(
                        containers=[_build_container(request, env_vars)],
                        volumes=_build_volumes(),
                        restart_policy="Never",
                        service_account_name="ml-training-operator",
                    ),
                ),
            ),
        )
        jobs.append(job)

    return jobs


def build_headless_service_spec(job_id: str) -> client.V1Service:
    """Build a headless Service spec for DDP worker discovery."""
    return client.V1Service(
        metadata=client.V1ObjectMeta(
            name=f"train-{job_id}-headless",
            labels={"app": "ml-training-worker", "job-id": job_id},
        ),
        spec=client.V1ServiceSpec(
            cluster_ip="None",
            selector={"app": "ml-training-worker", "job-id": job_id},
            ports=[client.V1ServicePort(name="ddp", port=DDP_PORT, target_port=DDP_PORT)],
        ),
    )
