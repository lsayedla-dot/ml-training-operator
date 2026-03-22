# Deploying ML Training Operator to Google Kubernetes Engine (GKE)

This guide walks through deploying the ML Training Operator end-to-end on a GKE Standard cluster, from cluster creation through monitoring and teardown.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Create a GKE Standard Cluster](#2-create-a-gke-standard-cluster)
3. [Create an Artifact Registry Repository](#3-create-an-artifact-registry-repository)
4. [Build and Push Docker Images](#4-build-and-push-docker-images)
5. [Apply Kustomize Manifests](#5-apply-kustomize-manifests)
6. [Verify the Deployment](#6-verify-the-deployment)
7. [Port-Forward the API and Test Endpoints](#7-port-forward-the-api-and-test-endpoints)
8. [Submit a Training Job via CLI](#8-submit-a-training-job-via-cli)
9. [Monitor with Prometheus and Grafana](#9-monitor-with-prometheus-and-grafana)
10. [Teardown](#10-teardown)

---

## 1. Prerequisites

Before you begin, make sure the following tools are installed and configured on your local machine.

| Tool | Minimum Version | Installation |
|------|-----------------|--------------|
| **gcloud CLI** | 450+ | [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install) |
| **kubectl** | 1.28+ | Installed via `gcloud components install kubectl` |
| **Docker** | 24+ | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) |
| **Python** | 3.11+ | Required for the `mltrain` CLI |

**GCP project requirements:**

- A GCP project with billing enabled.
- The following APIs must be active:
  - Kubernetes Engine API
  - Artifact Registry API
  - Compute Engine API

Enable them in one command:

```bash
gcloud services enable \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    compute.googleapis.com \
    --project=<YOUR_PROJECT_ID>
```

Authenticate and set your default project:

```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>
```

> **Tip:** Export your project ID as an environment variable to simplify the commands that follow.
>
> ```bash
> export GCP_PROJECT_ID="$(gcloud config get-value project)"
> ```

---

## 2. Create a GKE Standard Cluster

Create a single-node zonal cluster in `us-central1-a` using the cost-efficient `e2-medium` machine type:

```bash
gcloud container clusters create ml-training-cluster \
    --zone=us-central1-a \
    --num-nodes=1 \
    --machine-type=e2-medium \
    --disk-size=30 \
    --no-enable-autoupgrade \
    --project="${GCP_PROJECT_ID}"
```

This typically takes 3-5 minutes. Once the cluster is ready, fetch credentials so `kubectl` can communicate with it:

```bash
gcloud container clusters get-credentials ml-training-cluster \
    --zone=us-central1-a \
    --project="${GCP_PROJECT_ID}"
```

Confirm connectivity:

```bash
kubectl cluster-info
kubectl get nodes
```

You should see a single `Ready` node with the `e2-medium` machine type.

[screenshot placeholder: Terminal output showing `kubectl get nodes` with one Ready node]

---

## 3. Create an Artifact Registry Repository

Create a Docker repository in Artifact Registry to host the container images:

```bash
gcloud artifacts repositories create ml-training \
    --repository-format=docker \
    --location=us-central1 \
    --description="ML Training Operator container images" \
    --project="${GCP_PROJECT_ID}"
```

Configure Docker to authenticate against Artifact Registry:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
```

[screenshot placeholder: GCP Console showing the newly created Artifact Registry repository]

---

## 4. Build and Push Docker Images

From the repository root (`ml-training-operator/`), build both the API and worker images and push them to Artifact Registry.

Define the image tags:

```bash
export REGION="us-central1"
export API_IMAGE="${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/ml-training/ml-training-api:latest"
export WORKER_IMAGE="${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/ml-training/ml-training-worker:latest"
```

### Build and push the API image

```bash
docker build -f docker/api.Dockerfile -t "${API_IMAGE}" .
docker push "${API_IMAGE}"
```

### Build and push the Worker image

```bash
docker build -f docker/worker.Dockerfile -t "${WORKER_IMAGE}" .
docker push "${WORKER_IMAGE}"
```

> **Note:** The worker image includes PyTorch (CPU) and the nuScenes devkit, so it is larger than the API image. Expect the build to take several minutes.

Verify both images are present in the registry:

```bash
gcloud artifacts docker images list \
    us-central1-docker.pkg.dev/${GCP_PROJECT_ID}/ml-training \
    --project="${GCP_PROJECT_ID}"
```

[screenshot placeholder: Terminal output listing both pushed images in Artifact Registry]

---

## 5. Apply Kustomize Manifests

The project uses Kustomize overlays to manage environment-specific configuration. The `dev` overlay applies development resource limits and sets `LOG_LEVEL=DEBUG`.

Before applying, update the image references in the deployments to point to your Artifact Registry images:

```bash
cd k8s/overlays/dev/

kustomize edit set image \
    ml-training-api:latest="${API_IMAGE}" \
    ml-training-worker:latest="${WORKER_IMAGE}"

cd -
```

Apply the full stack:

```bash
kubectl apply -k k8s/overlays/dev/
```

This creates the following resources in the `ml-training` namespace:

| Resource | Name | Purpose |
|----------|------|---------|
| Namespace | `ml-training` | Isolates all operator resources |
| ServiceAccount + RBAC | `ml-training-operator` | Grants the controller permission to manage Jobs and Pods |
| ConfigMap | `ml-training-config` | Operator configuration (CPU/memory defaults, retry policy, log level) |
| PersistentVolumeClaim | `ml-training-db` | SQLite database storage for job metadata |
| Deployment | `ml-training-api` | FastAPI server (port 8000) |
| Deployment | `ml-training-controller` | Reconciliation loop that manages training Jobs |
| Service | `ml-training-api` | ClusterIP service routing port 80 to the API |
| ServiceMonitor | `ml-training-api` | Prometheus scrape config for `/metrics` |

[screenshot placeholder: Terminal output showing `kubectl apply` creating all resources]

---

## 6. Verify the Deployment

Wait for both deployments to roll out:

```bash
kubectl -n ml-training rollout status deployment/ml-training-api --timeout=120s
kubectl -n ml-training rollout status deployment/ml-training-controller --timeout=120s
```

Check that all pods are running:

```bash
kubectl get pods -n ml-training
```

Expected output:

```
NAME                                       READY   STATUS    RESTARTS   AGE
ml-training-api-xxxxxxxxx-xxxxx            1/1     Running   0          45s
ml-training-controller-xxxxxxxxx-xxxxx     1/1     Running   0          45s
```

If a pod is not in `Running` state, inspect it:

```bash
kubectl -n ml-training describe pod <POD_NAME>
kubectl -n ml-training logs <POD_NAME>
```

[screenshot placeholder: Terminal output showing all pods in Running state]

---

## 7. Port-Forward the API and Test Endpoints

Forward the API service to your local machine:

```bash
kubectl -n ml-training port-forward svc/ml-training-api 8000:80
```

In a separate terminal, test the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "healthy", ...}
```

Test the jobs listing endpoint:

```bash
curl http://localhost:8000/jobs
```

Expected response (empty list on a fresh deployment):

```json
[]
```

You can also verify the Prometheus metrics endpoint:

```bash
curl http://localhost:8000/metrics
```

[screenshot placeholder: Terminal showing successful curl responses from /health and /jobs]

---

## 8. Submit a Training Job via CLI

Install the `mltrain` CLI locally (requires Python 3.11+):

```bash
pip install -e .
```

With the port-forward still active, submit a training job:

```bash
mltrain --api-url http://localhost:8000 submit \
    --name "gke-test-resnet18" \
    --model resnet18 \
    --epochs 5 \
    --batch-size 16 \
    --lr 0.001 \
    --workers 1 \
    --cpu 2 \
    --memory 4Gi
```

List jobs to confirm it was accepted:

```bash
mltrain --api-url http://localhost:8000 list
```

Check the status of a specific job (replace `<JOB_ID>` with the actual ID from the submit output):

```bash
mltrain --api-url http://localhost:8000 status <JOB_ID>
```

View training logs:

```bash
mltrain --api-url http://localhost:8000 logs <JOB_ID>
```

To cancel a running job:

```bash
mltrain --api-url http://localhost:8000 cancel <JOB_ID>
```

[screenshot placeholder: CLI output showing a submitted job and its status progression]

---

## 9. Monitor with Prometheus and Grafana

### Install the Prometheus Stack

Deploy the `kube-prometheus-stack` via Helm, which includes Prometheus, Grafana, and the ServiceMonitor CRD:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install monitoring prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --wait
```

The `serviceMonitorSelectorNilUsesHelmValues=false` flag ensures Prometheus discovers the `ServiceMonitor` in the `ml-training` namespace.

### Verify Prometheus is Scraping Metrics

Port-forward to Prometheus:

```bash
kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090
```

Open [http://localhost:9090/targets](http://localhost:9090/targets) in your browser and confirm that the `ml-training/ml-training-api` target shows as **UP**.

[screenshot placeholder: Prometheus Targets page showing ml-training-api target as UP]

### Access Grafana

Port-forward to Grafana:

```bash
kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80
```

Open [http://localhost:3000](http://localhost:3000) and log in with the default credentials:

- **Username:** `admin`
- **Password:** `prom-operator`

### Import the ML Training Dashboard

1. Navigate to **Dashboards** > **Import**.
2. Upload the dashboard JSON file located at `k8s/base/prometheus/grafana-dashboard.json`.
3. Select the **Prometheus** data source when prompted.
4. Click **Import**.

The dashboard provides visibility into training job counts, durations, success/failure rates, and API request metrics.

[screenshot placeholder: Grafana dashboard showing ML Training Operator metrics panels]

---

## 10. Teardown

When you are finished, clean up all resources to avoid ongoing charges.

### Delete Kubernetes Resources

```bash
kubectl delete -k k8s/overlays/dev/ --ignore-not-found
```

### Delete the Prometheus Stack (if installed)

```bash
helm uninstall monitoring --namespace monitoring
kubectl delete namespace monitoring --ignore-not-found
```

### Delete the GKE Cluster

```bash
gcloud container clusters delete ml-training-cluster \
    --zone=us-central1-a \
    --project="${GCP_PROJECT_ID}" \
    --quiet
```

### Delete the Artifact Registry Repository

```bash
gcloud artifacts repositories delete ml-training \
    --location=us-central1 \
    --project="${GCP_PROJECT_ID}" \
    --quiet
```

> **Alternatively**, run the provided teardown script which handles all of the above:
>
> ```bash
> bash scripts/teardown_gke.sh
> ```

[screenshot placeholder: Terminal output confirming cluster and repository deletion]

---

## Quick Reference

| Action | Command |
|--------|---------|
| Deploy (scripted) | `bash scripts/deploy_gke.sh` |
| Deploy (manual) | `kubectl apply -k k8s/overlays/dev/` |
| Port-forward API | `kubectl -n ml-training port-forward svc/ml-training-api 8000:80` |
| Health check | `curl http://localhost:8000/health` |
| Submit job | `mltrain --api-url http://localhost:8000 submit --name test --model resnet18 --epochs 5` |
| List jobs | `mltrain --api-url http://localhost:8000 list` |
| View logs | `mltrain --api-url http://localhost:8000 logs <JOB_ID>` |
| Teardown (scripted) | `bash scripts/teardown_gke.sh` |
| Teardown (manual) | `gcloud container clusters delete ml-training-cluster --zone=us-central1-a --quiet` |
