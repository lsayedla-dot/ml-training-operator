"""Integration tests for the full job lifecycle."""

from __future__ import annotations


def test_job_lifecycle(api_client):
    """Submit job → status updates → completion flow."""
    # Step 1: Submit job
    create_resp = api_client.post(
        "/jobs",
        json={
            "name": "lifecycle-test",
            "model_type": "resnet18",
            "epochs": 3,
            "batch_size": 16,
            "num_workers": 1,
        },
    )
    assert create_resp.status_code == 201
    job = create_resp.json()
    job_id = job["id"]
    assert job["status"] in ("PENDING", "RUNNING")

    # Step 2: Job appears in list
    list_resp = api_client.get("/jobs")
    assert list_resp.status_code == 200
    job_ids = [j["id"] for j in list_resp.json()]
    assert job_id in job_ids

    # Step 3: Get job detail
    detail_resp = api_client.get(f"/jobs/{job_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["name"] == "lifecycle-test"

    # Step 4: Cancel job
    cancel_resp = api_client.delete(f"/jobs/{job_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "CANCELLED"

    # Step 5: Verify final state
    final_resp = api_client.get(f"/jobs/{job_id}")
    assert final_resp.json()["status"] == "CANCELLED"


def test_distributed_job_submission(api_client):
    """Submit a distributed job with multiple workers."""
    resp = api_client.post(
        "/jobs",
        json={
            "name": "distributed-lifecycle",
            "model_type": "resnet50",
            "epochs": 5,
            "num_workers": 4,
            "enable_optimization": True,
        },
    )
    assert resp.status_code == 201
    job = resp.json()
    assert job["num_workers"] == 4
    assert job["enable_optimization"] is True
