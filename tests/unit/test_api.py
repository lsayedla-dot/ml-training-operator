"""Tests for the Training Job API endpoints."""

from __future__ import annotations


def test_submit_job_valid(api_client):
    """POST /jobs with valid config returns 201 and job ID."""
    payload = {
        "name": "test-job-1",
        "model_type": "resnet18",
        "epochs": 5,
        "batch_size": 16,
    }
    resp = api_client.post("/jobs", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert data["name"] == "test-job-1"
    assert data["status"] in ("PENDING", "RUNNING")


def test_submit_job_invalid(api_client):
    """POST /jobs with invalid config returns 422."""
    resp = api_client.post("/jobs", json={"name": "", "epochs": 5})
    assert resp.status_code == 422


def test_list_jobs(api_client):
    """GET /jobs returns a list of jobs."""
    api_client.post("/jobs", json={"name": "list-test-1", "epochs": 3})
    api_client.post("/jobs", json={"name": "list-test-2", "epochs": 3})
    resp = api_client.get("/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 2


def test_list_jobs_with_filter(api_client):
    """GET /jobs?status=RUNNING returns filtered list."""
    api_client.post("/jobs", json={"name": "filter-test", "epochs": 3})
    resp = api_client.get("/jobs", params={"status": "RUNNING"})
    assert resp.status_code == 200
    data = resp.json()
    assert all(j["status"] == "RUNNING" for j in data)


def test_get_job_by_id(api_client):
    """GET /jobs/{id} returns job detail."""
    create_resp = api_client.post("/jobs", json={"name": "get-test", "epochs": 3})
    job_id = create_resp.json()["id"]

    resp = api_client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == job_id


def test_get_job_not_found(api_client):
    """GET /jobs/{nonexistent} returns 404."""
    resp = api_client.get("/jobs/nonexistent-id")
    assert resp.status_code == 404


def test_cancel_job(api_client):
    """DELETE /jobs/{id} cancels the job."""
    create_resp = api_client.post("/jobs", json={"name": "cancel-test", "epochs": 3})
    job_id = create_resp.json()["id"]

    resp = api_client.delete(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "CANCELLED"


def test_health_check(api_client):
    """GET /health returns 200 with status info."""
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "k8s_connected" in data
    assert "db_connected" in data
