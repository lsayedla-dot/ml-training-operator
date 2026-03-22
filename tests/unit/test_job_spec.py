"""Tests for K8s Job spec generation."""

from __future__ import annotations

from src.controller.job_spec import build_distributed_job_specs, build_single_job_spec


def test_generates_valid_job_spec(sample_job_request):
    """Generates a valid K8s Job spec from request."""
    job = build_single_job_spec(sample_job_request, "test-id-1")
    assert job.metadata.name.startswith("train-test-nuscenes-v1-")
    assert job.spec.template.spec.containers[0].image == "ml-training-worker:latest"


def test_resource_limits_set(sample_job_request):
    """Resource requests/limits are set correctly."""
    job = build_single_job_spec(sample_job_request, "test-id-2")
    container = job.spec.template.spec.containers[0]
    assert container.resources.requests["cpu"] == "2"
    assert container.resources.requests["memory"] == "4Gi"
    assert container.resources.limits["cpu"] == "2"
    assert container.resources.limits["memory"] == "4Gi"


def test_env_vars_passed(sample_job_request):
    """Environment variables are passed through."""
    job = build_single_job_spec(sample_job_request, "test-id-3")
    container = job.spec.template.spec.containers[0]
    env_names = {e.name for e in container.env}
    assert "MODEL_TYPE" in env_names
    assert "EPOCHS" in env_names
    assert "BATCH_SIZE" in env_names
    assert "LEARNING_RATE" in env_names


def test_volume_mounts_present(sample_job_request):
    """Volume mounts for checkpoints are present."""
    job = build_single_job_spec(sample_job_request, "test-id-4")
    container = job.spec.template.spec.containers[0]
    mount_names = {m.name for m in container.volume_mounts}
    assert "checkpoints" in mount_names
    assert "data" in mount_names


def test_labels_include_job_id(sample_job_request):
    """Labels include job-id."""
    job = build_single_job_spec(sample_job_request, "test-id-5")
    assert job.metadata.labels["job-id"] == "test-id-5"


def test_restart_policy_is_never(sample_job_request):
    """Restart policy is Never."""
    job = build_single_job_spec(sample_job_request, "test-id-6")
    assert job.spec.template.spec.restart_policy == "Never"


def test_distributed_specs(distributed_job_request):
    """Distributed job creates N jobs with correct rank env vars."""
    jobs = build_distributed_job_specs(distributed_job_request, "dist-id-1")
    assert len(jobs) == 4

    for rank, job in enumerate(jobs):
        container = job.spec.template.spec.containers[0]
        env_dict = {e.name: e.value for e in container.env}
        assert env_dict["RANK"] == str(rank)
        assert env_dict["WORLD_SIZE"] == "4"
        assert "MASTER_ADDR" in env_dict
        assert "MASTER_PORT" in env_dict
