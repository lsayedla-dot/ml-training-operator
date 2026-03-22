"""Thin wrapper around the Kubernetes Python client."""

from __future__ import annotations

import structlog
from kubernetes import client, config
from kubernetes.client import (
    V1DeleteOptions,
    V1Job,
    V1Service,
)
from kubernetes.client.exceptions import ApiException

logger = structlog.get_logger()

DEFAULT_NAMESPACE = "ml-training"


class K8sClient:
    """Wrapper around Kubernetes API for managing training jobs."""

    def __init__(self, namespace: str = DEFAULT_NAMESPACE) -> None:
        self.namespace = namespace
        self._batch_api: client.BatchV1Api | None = None
        self._core_api: client.CoreV1Api | None = None
        self._connected = False

    def connect(self) -> bool:
        """Initialize Kubernetes API clients."""
        try:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            self._batch_api = client.BatchV1Api()
            self._core_api = client.CoreV1Api()
            self._connected = True
            logger.info("k8s_connected", namespace=self.namespace)
            return True
        except Exception as e:
            logger.warning("k8s_connection_failed", error=str(e))
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        """Check if connected to K8s."""
        return self._connected

    def create_namespaced_job(self, job: V1Job) -> V1Job:
        """Create a K8s Job in the configured namespace."""
        if not self._batch_api:
            raise RuntimeError("K8s client not connected")
        try:
            result = self._batch_api.create_namespaced_job(namespace=self.namespace, body=job)
            logger.info(
                "k8s_job_created",
                job_name=result.metadata.name,
                namespace=self.namespace,
            )
            return result
        except ApiException as e:
            logger.error("k8s_job_create_failed", error=str(e), status=e.status)
            raise

    def read_namespaced_job_status(self, job_name: str) -> V1Job:
        """Read the status of a K8s Job."""
        if not self._batch_api:
            raise RuntimeError("K8s client not connected")
        return self._batch_api.read_namespaced_job_status(name=job_name, namespace=self.namespace)

    def delete_namespaced_job(self, job_name: str) -> None:
        """Delete a K8s Job and its pods."""
        if not self._batch_api:
            raise RuntimeError("K8s client not connected")
        try:
            self._batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=V1DeleteOptions(propagation_policy="Foreground"),
            )
            logger.info("k8s_job_deleted", job_name=job_name)
        except ApiException as e:
            if e.status == 404:
                logger.warning("k8s_job_not_found", job_name=job_name)
            else:
                raise

    def read_namespaced_pod_log(self, pod_name: str) -> str:
        """Read logs from a pod."""
        if not self._core_api:
            raise RuntimeError("K8s client not connected")
        try:
            return self._core_api.read_namespaced_pod_log(name=pod_name, namespace=self.namespace)
        except ApiException as e:
            if e.status == 404:
                return ""
            raise

    def list_pods_for_job(self, job_name: str) -> list:
        """List pods belonging to a K8s Job."""
        if not self._core_api:
            raise RuntimeError("K8s client not connected")
        pods = self._core_api.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"job-name={job_name}",
        )
        return pods.items

    def create_headless_service(self, job_id: str, num_workers: int) -> V1Service:
        """Create a headless Service for DDP worker discovery."""
        if not self._core_api:
            raise RuntimeError("K8s client not connected")
        service = V1Service(
            metadata=client.V1ObjectMeta(
                name=f"train-{job_id}-headless",
                namespace=self.namespace,
                labels={"app": "ml-training-worker", "job-id": job_id},
            ),
            spec=client.V1ServiceSpec(
                cluster_ip="None",
                selector={"app": "ml-training-worker", "job-id": job_id},
                ports=[client.V1ServicePort(name="ddp", port=29500, target_port=29500)],
            ),
        )
        result = self._core_api.create_namespaced_service(namespace=self.namespace, body=service)
        logger.info("k8s_headless_service_created", job_id=job_id, workers=num_workers)
        return result

    def delete_headless_service(self, job_id: str) -> None:
        """Delete the headless Service for a distributed job."""
        if not self._core_api:
            raise RuntimeError("K8s client not connected")
        svc_name = f"train-{job_id}-headless"
        try:
            self._core_api.delete_namespaced_service(name=svc_name, namespace=self.namespace)
            logger.info("k8s_headless_service_deleted", job_id=job_id)
        except ApiException as e:
            if e.status == 404:
                logger.warning("k8s_headless_service_not_found", job_id=job_id)
            else:
                raise
