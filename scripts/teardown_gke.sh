#!/bin/bash
# Teardown GKE cluster and clean up resources
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
ZONE="us-central1-a"
CLUSTER_NAME="ml-training-cluster"
REPO_NAME="ml-training"
REGION="us-central1"

echo "==> Tearing down ML Training Operator..."
echo "    Project: ${PROJECT_ID}"

# Delete K8s resources first
echo "==> Deleting Kubernetes resources..."
kubectl delete -k k8s/overlays/dev/ --ignore-not-found 2>/dev/null || true

# Delete GKE cluster
echo "==> Deleting GKE cluster..."
gcloud container clusters delete "${CLUSTER_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --quiet || echo "Cluster may not exist"

# Delete Artifact Registry images
echo "==> Cleaning up Artifact Registry..."
gcloud artifacts repositories delete "${REPO_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet 2>/dev/null || echo "Repository may not exist"

echo ""
echo "==> Teardown complete!"
