#!/bin/bash
# Deploy ML Training Operator to GKE
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
REGION="us-central1"
ZONE="${REGION}-a"
CLUSTER_NAME="ml-training-cluster"
REPO_NAME="ml-training"
API_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/ml-training-api:latest"
WORKER_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/ml-training-worker:latest"

echo "==> Project: ${PROJECT_ID}"
echo "==> Zone: ${ZONE}"

# Step 1: Create GKE cluster
echo "==> Creating GKE cluster..."
gcloud container clusters create "${CLUSTER_NAME}" \
    --zone="${ZONE}" \
    --num-nodes=1 \
    --machine-type=e2-medium \
    --disk-size=30 \
    --no-enable-autoupgrade \
    --project="${PROJECT_ID}" || echo "Cluster may already exist"

# Step 2: Get credentials
echo "==> Getting cluster credentials..."
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}"

# Step 3: Create Artifact Registry repository
echo "==> Creating Artifact Registry repository..."
gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" 2>/dev/null || echo "Repository may already exist"

# Step 4: Configure Docker auth
echo "==> Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Step 5: Build and push images
echo "==> Building and pushing API image..."
docker build -f docker/api.Dockerfile -t "${API_IMAGE}" .
docker push "${API_IMAGE}"

echo "==> Building and pushing Worker image..."
docker build -f docker/worker.Dockerfile -t "${WORKER_IMAGE}" .
docker push "${WORKER_IMAGE}"

# Step 6: Update image references in kustomization
echo "==> Applying Kubernetes manifests..."
kubectl apply -k k8s/overlays/dev/

# Step 7: Wait for rollout
echo "==> Waiting for API deployment..."
kubectl -n ml-training rollout status deployment/ml-training-api --timeout=120s

echo "==> Waiting for Controller deployment..."
kubectl -n ml-training rollout status deployment/ml-training-controller --timeout=120s

# Step 8: Port forward
echo ""
echo "============================================"
echo "  ML Training Operator deployed successfully!"
echo "============================================"
echo ""
echo "To access the API:"
echo "  kubectl -n ml-training port-forward svc/ml-training-api 8000:80"
echo ""
echo "Then:"
echo "  curl http://localhost:8000/health"
echo "  mltrain --api-url http://localhost:8000 list"
echo ""
