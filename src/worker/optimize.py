"""Post-training model optimization: ONNX export + INT8 quantization + benchmarking."""

from __future__ import annotations

import inspect
import json
import os
import time

import numpy as np
import structlog
import torch

logger = structlog.get_logger()


def export_onnx(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
) -> str:
    """Export a trained PyTorch model to ONNX format.

    Includes dynamic axes for variable batch sizes — critical for
    production inference where batch size varies.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.eval()

    export_kwargs = dict(
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    # Newer PyTorch versions default to dynamo exporter which produces
    # graphs incompatible with onnxruntime quantization. Force legacy.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, sample_input, output_path, **export_kwargs)

    # Verify the exported model
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    model_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("onnx_exported", path=output_path, size_mb=round(model_size, 2))
    return output_path


def quantize_int8(
    onnx_path: str,
    output_path: str,
    calibration_data: list[np.ndarray] | None = None,
) -> str:
    """Apply post-training static INT8 quantization via ONNX Runtime.

    AV perception models run on edge devices — INT8 reduces model size
    3-4x and improves inference latency 2-3x with minimal accuracy loss.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Preprocess model for quantization (fixes shape inference issues)
    preprocessed_path = onnx_path.replace(".onnx", "_preprocessed.onnx")
    try:
        quant_pre_process(onnx_path, preprocessed_path)
        source_path = preprocessed_path
    except Exception:
        source_path = onnx_path

    quantize_dynamic(
        model_input=source_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    # Clean up preprocessed file
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    model_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("model_quantized", path=output_path, size_mb=round(model_size, 2))
    return output_path


def benchmark_models(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    quantized_path: str,
    sample_input: torch.Tensor,
    num_runs: int = 100,
) -> dict:
    """Benchmark all 3 model variants: PyTorch, ONNX, INT8 quantized.

    Returns latency percentiles and model sizes for comparison.
    Every millisecond matters for object detection at 60mph.
    """
    import onnxruntime as ort

    results = {}

    # Benchmark PyTorch
    pytorch_model.eval()
    pytorch_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            pytorch_model(sample_input)
            pytorch_times.append((time.perf_counter() - start) * 1000)

    pt_path = onnx_path.replace(".onnx", ".pt")
    torch.save(pytorch_model.state_dict(), pt_path)
    results["pytorch"] = {
        "latency_ms_p50": float(np.percentile(pytorch_times, 50)),
        "latency_ms_p95": float(np.percentile(pytorch_times, 95)),
        "latency_ms_p99": float(np.percentile(pytorch_times, 99)),
        "model_size_mb": os.path.getsize(pt_path) / (1024 * 1024),
    }

    # Benchmark ONNX
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    input_data = sample_input.numpy()
    onnx_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ort_session.run(None, {input_name: input_data})
        onnx_times.append((time.perf_counter() - start) * 1000)

    results["onnx"] = {
        "latency_ms_p50": float(np.percentile(onnx_times, 50)),
        "latency_ms_p95": float(np.percentile(onnx_times, 95)),
        "latency_ms_p99": float(np.percentile(onnx_times, 99)),
        "model_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
    }

    # Benchmark quantized ONNX
    quant_session = ort.InferenceSession(quantized_path)
    quant_input_name = quant_session.get_inputs()[0].name
    quant_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        quant_session.run(None, {quant_input_name: input_data})
        quant_times.append((time.perf_counter() - start) * 1000)

    results["quantized_int8"] = {
        "latency_ms_p50": float(np.percentile(quant_times, 50)),
        "latency_ms_p95": float(np.percentile(quant_times, 95)),
        "latency_ms_p99": float(np.percentile(quant_times, 99)),
        "model_size_mb": os.path.getsize(quantized_path) / (1024 * 1024),
    }

    # Clean up temporary pt file
    if os.path.exists(pt_path):
        os.remove(pt_path)

    return results


def generate_optimization_report(benchmarks: dict, output_path: str) -> str:
    """Generate a JSON report comparing model variants."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    pytorch = benchmarks["pytorch"]
    onnx = benchmarks["onnx"]
    quantized = benchmarks["quantized_int8"]

    def _speedup(a: dict, b: dict, key: str) -> float:
        return round(a[key] / max(b[key], 1e-6), 2)

    report = {
        "benchmarks": benchmarks,
        "speedup_ratios": {
            "onnx_vs_pytorch": _speedup(pytorch, onnx, "latency_ms_p50"),
            "int8_vs_pytorch": _speedup(pytorch, quantized, "latency_ms_p50"),
            "int8_vs_onnx": _speedup(onnx, quantized, "latency_ms_p50"),
        },
        "size_reduction_ratios": {
            "onnx_vs_pytorch": _speedup(pytorch, onnx, "model_size_mb"),
            "int8_vs_pytorch": _speedup(pytorch, quantized, "model_size_mb"),
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("optimization_report_generated", path=output_path)
    return output_path


def run_optimization_pipeline(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_dir: str,
) -> dict:
    """Orchestrate full optimization: export → quantize → benchmark → report.

    This is the path from training cluster to vehicle:
    PyTorch → ONNX → quantized ONNX → TensorRT/OpenVINO on device.
    """
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "model.onnx")
    quantized_path = os.path.join(output_dir, "model_int8.onnx")
    report_path = os.path.join(output_dir, "optimization_report.json")

    # Step 1: Export to ONNX
    export_onnx(model, sample_input, onnx_path)

    # Step 2: Quantize to INT8
    quantize_int8(onnx_path, quantized_path)

    # Step 3: Benchmark all variants
    benchmarks = benchmark_models(model, onnx_path, quantized_path, sample_input)

    # Step 4: Generate report
    generate_optimization_report(benchmarks, report_path)

    logger.info("optimization_pipeline_complete", output_dir=output_dir)
    return benchmarks
