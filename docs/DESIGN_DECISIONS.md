# Design Decisions

This document captures the key architectural and tooling decisions made during the design of the ML Training Job Operator. Each decision is presented with the alternatives we evaluated, the rationale for our choice, and the trade-offs we accepted.

---

## 1. Custom Operator over Argo Workflows / Kubeflow Training Operator

**Decision:** Build a purpose-built Kubernetes operator from scratch rather than adopting Argo Workflows or the Kubeflow Training Operator.

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **Custom Operator** | Full control over reconciliation logic; minimal CRD surface area; demonstrates deep understanding of K8s primitives (informers, work queues, leader election); tailored exactly to our training semantics | Requires implementing controller plumbing ourselves; no community-maintained UI or plugin ecosystem out of the box |
| **Argo Workflows** | Mature DAG execution engine; built-in UI for workflow visualization; large community and plugin ecosystem | Heavy CRD footprint (Workflow, WorkflowTemplate, CronWorkflow, etc.); pulls in a full workflow engine when we only need job orchestration; opaque retry and scheduling logic that is difficult to customize for ML-specific semantics |
| **Kubeflow Training Operator** | Purpose-built for ML training (TFJob, PyTorchJob, MPIJob); handles multi-worker coordination natively | Requires the broader Kubeflow ecosystem or significant effort to run standalone; large dependency surface; abstractions are generalized across frameworks, making it harder to optimize for a single-framework (PyTorch) workflow |

### Why

The operator pattern is the idiomatic way to extend Kubernetes for domain-specific workloads. By building our own controller, we keep the CRD schema minimal -- a single `TrainingJob` resource -- and own every line of the reconciliation loop. This makes debugging straightforward: there is no framework-internal state machine to reverse-engineer when a job stalls. It also serves as a concrete demonstration of Kubernetes internals knowledge: writing a controller that watches resources, computes diffs, and drives the cluster toward a desired state is the core skill behind any production operator.

### Trade-off

We accept the upfront cost of writing and maintaining controller boilerplate (client setup, error handling, status patching) in exchange for a smaller dependency footprint and complete control over scheduling, retry, and observability semantics. If the project later requires DAG-style pipelines or multi-framework support, we would revisit this decision.

---

## 2. Kustomize over Helm for Manifest Management

**Decision:** Use Kustomize with base/overlay directories instead of Helm charts for all Kubernetes manifest generation.

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **Kustomize** | Native to `kubectl` (no additional tooling); operates on plain YAML with strategic merge patches; overlays map directly to environments (dev, staging, prod); no template syntax to learn or debug | Limited conditional logic; no built-in package registry; less ecosystem momentum for third-party chart distribution |
| **Helm** | Powerful Go templating for parameterized manifests; large public chart repository; built-in release management with rollback | Template engine introduces a DSL layer over YAML that is notoriously difficult to debug (`{{ if }}`, `{{ range }}`, whitespace control); values files can grow unwieldy; tiller (v2) had significant security concerns, and even v3 adds client-side complexity |
| **Raw YAML + envsubst** | Zero tooling dependencies; trivially understandable | No merge semantics; environment-specific changes require full file duplication; error-prone for large manifests |

### Why

Kustomize aligns with the principle of least abstraction. Our manifests are plain, valid Kubernetes YAML at every layer. A developer can `kubectl apply -k overlays/dev/` without installing anything beyond kubectl. Overlays for resource limits, replica counts, and namespace bindings are expressed as patches rather than template variables, which means every intermediate artifact is itself a valid manifest that can be linted, diffed, and applied independently.

### Trade-off

We give up Helm's conditional logic and its package distribution model. If we needed to publish the operator as a reusable community chart, Helm would be the better choice. For an internal, single-project operator where clarity and auditability matter more than parameterization breadth, Kustomize is the simpler path.

---

## 3. Kubernetes Jobs over Deployments for Training Runs

**Decision:** Model each training run as a Kubernetes `Job` (or `Job` with indexed completions for distributed runs) rather than a `Deployment`.

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **Kubernetes Job** | Correct semantic for batch/run-to-completion workloads; built-in completion tracking and failure counting; `backoffLimit` provides basic retry; pods are not restarted after successful exit | No long-lived endpoint; requires a new Job resource per training run |
| **Deployment** | Familiar abstraction; built-in rolling updates; always-on replicas with self-healing | Designed for long-running services, not batch work; a completed training process would be restarted by the replica controller; no native concept of "done"; `restartPolicy: Always` fights against the desired run-to-completion behavior |
| **Bare Pods** | Simplest possible unit; no controller overhead | No retry on failure; no completion tracking; not rescheduled if a node dies; operationally fragile |

### Why

A training run has a clear start and end. It reads data, computes gradients for N epochs, saves a checkpoint, and exits. The Kubernetes Job resource was designed precisely for this lifecycle. Using a Deployment would require fighting its core assumption -- that the desired state is a stable set of running replicas -- by adding shutdown hooks, completion signals, and external coordination to prevent restart loops. Jobs give us completion tracking, configurable parallelism, and failure policies for free.

### Trade-off

Jobs are immutable after creation (you cannot update a Job's pod template). If a training run needs parameter changes mid-flight, we must cancel and create a new Job. For our use case -- where hyperparameters are fixed at submission time -- this is not a limitation.

---

## 4. SQLite over PostgreSQL for Job Metadata Storage

**Decision:** Store job metadata (submission records, status history, hyperparameter snapshots) in an embedded SQLite database rather than a managed PostgreSQL instance.

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **SQLite** | Embedded, zero-ops; single file on a PersistentVolume; no network round-trips; no connection pool management; trivial backup (copy the file); sufficient for the expected write throughput of a training orchestrator | Single-writer concurrency model; not suitable for horizontally scaled API servers; data lives on a single volume (no built-in replication) |
| **PostgreSQL** | Full ACID with MVCC; handles concurrent writers; rich ecosystem of extensions (pg_stat, pg_trgm); production-proven at scale | Requires a separate StatefulSet or managed service; connection string management; backup/restore tooling; operational overhead for what is fundamentally a low-write metadata store |
| **etcd (via K8s API annotations/status)** | No external storage at all; metadata lives with the CRD objects | etcd is not a general-purpose database; large status payloads degrade API server performance; no query capability; 1.5MB object size limit |

### Why

A training job orchestrator submits jobs at human timescales -- tens to hundreds per day, not thousands per second. SQLite handles this write volume effortlessly. By embedding the database, we eliminate an entire infrastructure dependency: no StatefulSet to manage, no credentials to rotate, no connection pool to tune. The database file sits on a PersistentVolumeClaim, and backup is a file copy. This keeps the operator's deployment footprint minimal and its failure domain small.

### Trade-off

If the operator needs to scale to multiple active-active controller replicas, SQLite's single-writer model becomes a bottleneck. At that point, migrating to PostgreSQL (or CockroachDB for multi-region) would be necessary. We also accept that queries against SQLite lack the advanced indexing and full-text search capabilities of PostgreSQL, though our query patterns (lookup by job ID, filter by status) are simple enough that this is not a concern.

---

## 5. Controller Polling over Kubernetes Watch API

**Decision:** The controller reconciliation loop polls the Kubernetes API at a configurable interval rather than using the Watch API (informers with event handlers).

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **Polling (periodic list + reconcile)** | Simple mental model: "every N seconds, list all TrainingJobs and reconcile each one"; inherently resistant to missed events; no complex bookmark/resourceVersion management; easy to reason about in tests | Higher API server load than watches at scale; reconciliation latency is bounded by the poll interval; redundant work when nothing has changed |
| **Watch API (informers)** | Event-driven, near-zero latency on state changes; efficient use of API server resources via long-lived HTTP streams; the idiomatic pattern for production operators | Watch connections can break silently; requires correct handling of bookmark events, 410 Gone responses, and re-list fallback; subtle bugs around stale cache reads; significantly more complex to implement correctly and to test |
| **Hybrid (informer + periodic resync)** | Best of both worlds: immediate reaction to events plus periodic consistency checks | Most complex implementation; two code paths to maintain and test; resync interval tuning adds operational surface |

### Why

Correctness is easier to achieve with polling. A poll-based loop is a pure function: given the current state of the world (list of TrainingJob resources + their associated Jobs/Pods), compute the desired actions and execute them. There are no edge cases around missed watch events, no need to handle `BOOKMARK` or `GONE` responses, and no shared informer cache to reason about. For an operator managing tens to low hundreds of training jobs, the additional API server load from periodic listing is negligible. The poll interval (default: 30 seconds) provides a clear, tunable knob for reconciliation latency.

### Trade-off

We accept higher reconciliation latency (up to the poll interval) compared to a watch-based controller, which would react within milliseconds of a state change. For training jobs that run for minutes to hours, a 30-second reconciliation delay is immaterial. If the operator later manages thousands of concurrent jobs, the List calls would become expensive and a migration to informer-based watching would be warranted.

---

## 6. Retry Logic at Controller Level, Not Kubernetes Job Level

**Decision:** Implement retry, backoff, and dead-lettering logic in the operator's reconciliation loop rather than relying on the Kubernetes Job's built-in `backoffLimit` and `restartPolicy`.

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **Controller-level retry** | Full control over delay calculation (exponential, jittered, capped); structured logging at each retry with failure reason; ability to route permanently-failed jobs to a dead-letter queue or status; can implement circuit-breaker patterns; retry policy is defined in the TrainingJob CRD, not scattered across Job specs | More code to write and maintain; the controller must track retry state (count, next attempt time) in the CRD status or SQLite |
| **Kubernetes Job `backoffLimit`** | Zero application code; Kubernetes handles pod restart with exponential backoff natively | Backoff timing is opaque and not configurable beyond the limit count; no structured logging of retry reasons; no dead-letter concept; `backoffLimit` interacts with `restartPolicy` in non-obvious ways (e.g., container-level restart vs. pod-level restart); difficult to distinguish transient OOM from permanent data errors |
| **External retry system (e.g., Temporal, Celery)** | Battle-tested retry semantics; built-in dead-letter queues; visibility into retry state | Introduces a heavyweight external dependency; moves coordination logic outside the Kubernetes control plane; increases operational complexity |

### Why

Training job failures are not homogeneous. An OOM kill on a large batch size is worth retrying with a smaller batch. A data loader crash due to a corrupted file is not worth retrying at all. A transient NFS timeout is worth retrying after a brief delay. Kubernetes' `backoffLimit` treats all failures identically: it increments a counter and waits. By owning the retry loop, we can inspect the exit code and pod events, classify the failure, adjust retry strategy accordingly, and emit structured logs that make post-mortem analysis straightforward. Jobs that exhaust their retry budget are marked as `DeadLettered` in the CRD status, making them easy to query and investigate.

### Trade-off

We take on the complexity of maintaining retry state and ensuring it survives controller restarts (persisted in the CRD status subresource). We also must be careful to avoid conflicting with any residual Kubernetes-level retry behavior -- we set `backoffLimit: 0` on created Jobs to ensure the controller is the sole retry authority.

---

## 7. nuScenes Dataset over Synthetic Data or CIFAR

**Decision:** Use the nuScenes dataset as the primary data source for training and validation rather than synthetic datasets or standard ML benchmarks like CIFAR-10/100 or ImageNet.

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **nuScenes** | Real-world autonomous vehicle dataset created by Motional; multi-modal (camera, LiDAR, radar); includes 3D bounding box annotations for 23 object classes; demonstrates domain awareness and familiarity with AV data pipelines | Large dataset (~300GB full); requires a data loader that handles the nuScenes schema (scenes, samples, annotations); not as universally familiar as CIFAR or ImageNet |
| **CIFAR-10/100** | Tiny (163MB); universally understood; trivial data loading with `torchvision.datasets`; fast iteration cycles | Completely unrelated to autonomous driving; 32x32 images are meaningless for object detection; using it would signal a toy project rather than domain expertise |
| **Synthetic data (e.g., CARLA-generated)** | Unlimited volume; perfect annotations; configurable scenarios | Requires running a simulator or hosting pre-generated data; domain gap between synthetic and real sensor data; does not demonstrate familiarity with production AV datasets |
| **ImageNet** | Large-scale, well-understood benchmark; extensive pretrained model ecosystem | Classification-only (no 3D bounding boxes); 150GB+ download; not specific to autonomous driving |

### Why

nuScenes is Motional's own dataset, released to the research community. Using it signals direct familiarity with the data formats, coordinate systems, and annotation conventions that a Motional engineering team works with daily. The dataset's multi-modal nature (camera images, LiDAR point clouds, radar returns) exercises realistic data loading pipelines -- handling large binary blobs, spatial transformations, and sensor synchronization -- that a production training operator would need to support. It also makes the project's object detection task meaningful: we are training a model that could, in principle, detect pedestrians, vehicles, and cyclists from real sensor data, not classifying 32x32 thumbnails of cats and dogs.

### Trade-off

nuScenes is large and its data loader is more complex than a one-line `torchvision.datasets.CIFAR10()` call. We mitigate this by supporting a `mini` split (roughly 4GB) for development and CI, and only using the full dataset for end-to-end validation. The added complexity is justified by the domain relevance.

---

## 8. PyTorch DDP with Gloo Backend over Horovod / DeepSpeed

**Decision:** Use PyTorch's native Distributed Data Parallel (DDP) with the Gloo communication backend for multi-worker training, with worker discovery via a Kubernetes headless Service.

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **PyTorch DDP + Gloo** | Native to PyTorch (no additional library); Gloo runs on CPU and does not require NVIDAI NCCL or InfiniBand; headless Service for `MASTER_ADDR` discovery is a standard Kubernetes pattern; well-documented and broadly understood | Gloo is slower than NCCL for GPU-to-GPU communication; DDP requires all workers to have identical model architecture and batch size |
| **Horovod** | Framework-agnostic (TensorFlow, PyTorch, MXNet); uses MPI for communication, which is well-optimized for HPC clusters; ring-allreduce is bandwidth-optimal | Adds an external dependency (horovod, MPI, often requires custom Docker images with MPI libraries); MPI-based launching (`horovodrun`, `mpirun`) does not map cleanly to Kubernetes Job semantics without an MPI Operator; debugging MPI errors in containers is painful |
| **DeepSpeed** | ZeRO optimizer stages dramatically reduce memory footprint; built-in mixed precision, gradient accumulation, and pipeline parallelism; excellent for very large models | Heavyweight dependency; configuration complexity (ZeRO stages, offloading, activation checkpointing); overkill for models that fit in single-GPU memory; tight coupling to specific PyTorch versions |
| **PyTorch DDP + NCCL** | Fastest GPU-to-GPU collective operations; leverages NVLink/NVSwitch where available | Requires NVIDIA GPUs and the NCCL library; does not work on CPU-only clusters; sensitive to network topology and CUDA driver versions |

### Why

DDP is PyTorch's first-class distributed training API. It wraps the model in a thin communication layer that synchronizes gradients after each backward pass, with no changes to the training loop beyond initialization. The Gloo backend works on commodity hardware (including CPU-only nodes), making it suitable for development clusters that may not have GPUs. Worker discovery follows a clean Kubernetes pattern: a headless Service (ClusterIP: None) exposes stable DNS records for each worker pod, and the rank-0 pod's hostname serves as `MASTER_ADDR`. This is the same pattern used by StatefulSets and is well-understood by Kubernetes operators. No MPI daemon, no SSH key distribution, no custom launcher -- just environment variables and DNS.

### Trade-off

Gloo's communication performance is lower than NCCL for GPU workloads. In a production setting with multi-GPU nodes connected via NVLink, switching to the NCCL backend would be a one-line change (`init_process_group(backend="nccl")`). The architecture -- headless Service, environment-variable-based discovery, DDP wrapper -- remains identical regardless of backend. We optimize for portability and simplicity now, with a clear upgrade path to NCCL when GPU infrastructure is available.

---

## 9. ONNX Export with INT8 Quantization for Model Serving

**Decision:** Export trained PyTorch models to ONNX format and apply INT8 post-training quantization as the standard path to deployment-ready artifacts.

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **ONNX + INT8 quantization** | Framework-agnostic inference format; INT8 reduces model size by ~4x and increases throughput on edge accelerators; ONNX Runtime supports CPU, GPU, TensorRT, and edge devices; quantization is critical for real-time AV inference | Quantization can degrade accuracy (typically 0.5-2% mAP loss); ONNX export requires careful handling of dynamic shapes and custom ops; INT8 calibration requires a representative dataset |
| **TorchScript (JIT)** | Native PyTorch export; no framework conversion; supports dynamic control flow via tracing or scripting | Tied to PyTorch runtime; limited hardware accelerator support compared to ONNX; not as widely adopted for edge deployment |
| **TensorRT direct** | Maximum inference performance on NVIDIA hardware; INT8 and FP16 optimization built-in | NVIDIA-only; not portable to non-NVIDIA edge accelerators; tight coupling to specific GPU architectures; conversion can be fragile for complex models |
| **FP32 model serving (no quantization)** | No accuracy loss; simplest pipeline; no calibration step | 4x larger model; 2-4x slower inference; unacceptable for real-time object detection on edge compute where latency budgets are measured in single-digit milliseconds |

### Why

Autonomous vehicles run inference on edge compute platforms (e.g., NVIDIA Orin, Qualcomm SA8650) where every millisecond of latency matters. At 60 mph, a vehicle travels approximately 27 meters per second -- a 10ms inference delay means the perception system is "blind" for 27 centimeters of travel. INT8 quantization typically delivers a 2-4x speedup over FP32 on edge accelerators while maintaining acceptable accuracy for safety-critical object detection tasks (validated through per-class mAP analysis on the calibration set). ONNX provides the portability to target multiple hardware backends without rewriting the inference pipeline for each one.

The export pipeline is integrated into the operator: when a TrainingJob completes successfully, the controller can trigger an export step that converts the final checkpoint to ONNX, runs INT8 calibration against a held-out calibration split, validates quantized model accuracy against a minimum mAP threshold, and writes the resulting artifact to the model registry.

### Trade-off

Post-training quantization is simpler but less accurate than quantization-aware training (QAT), which fine-tunes the model with simulated quantization during training. We start with post-training quantization for its simplicity and move to QAT if the accuracy loss exceeds acceptable thresholds for any safety-critical object class (particularly pedestrians and cyclists, where detection failures have the highest consequence). The ONNX export also requires maintaining compatibility between the PyTorch model's operator set and the ONNX opset version, which adds a validation step to the CI pipeline.

---

## Summary

| # | Decision | Key Driver |
|---|---|---|
| 1 | Custom operator | Minimal dependencies, full control, demonstrates K8s expertise |
| 2 | Kustomize | Plain YAML, no template DSL, native to kubectl |
| 3 | K8s Jobs | Correct abstraction for run-to-completion workloads |
| 4 | SQLite | Zero-ops embedded storage for low-write metadata |
| 5 | Polling reconciliation | Simpler correctness model than watch-based controllers |
| 6 | Controller-level retry | Failure classification, structured logging, dead-lettering |
| 7 | nuScenes dataset | Domain relevance to autonomous vehicle perception |
| 8 | PyTorch DDP + Gloo | Native distributed training, standard K8s discovery pattern |
| 9 | ONNX + INT8 | Edge deployment readiness for real-time AV inference |

These decisions collectively optimize for **operational simplicity**, **Kubernetes-native patterns**, and **autonomous vehicle domain relevance**, while maintaining clear upgrade paths as requirements evolve.
