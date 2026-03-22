"""Tests for post-training model optimization pipeline."""

from __future__ import annotations

import json
import os

import pytest
import torch

try:
    import torchvision  # noqa: F401

    from src.worker.model import AVObjectClassifier

    _TORCHVISION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _TORCHVISION_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision not available")


@pytest.fixture
def trained_model():
    """A simple model for optimization testing."""
    return AVObjectClassifier(model_type="resnet18", pretrained=False)


@pytest.fixture
def sample_input():
    """Sample input tensor."""
    return torch.randn(1, 3, 224, 224)


def test_onnx_export(trained_model, sample_input, tmp_path):
    """ONNX export produces a valid .onnx file."""
    from src.worker.optimize import export_onnx

    output_path = str(tmp_path / "model.onnx")
    export_onnx(trained_model, sample_input, output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    # Verify it's valid ONNX
    import onnx

    model = onnx.load(output_path)
    onnx.checker.check_model(model)


def test_onnx_dynamic_axes(trained_model, sample_input, tmp_path):
    """Exported ONNX model accepts variable batch sizes."""
    import onnxruntime as ort

    from src.worker.optimize import export_onnx

    output_path = str(tmp_path / "model.onnx")
    export_onnx(trained_model, sample_input, output_path)

    session = ort.InferenceSession(output_path)
    input_name = session.get_inputs()[0].name

    # Test with batch size 1
    out1 = session.run(None, {input_name: torch.randn(1, 3, 224, 224).numpy()})
    assert out1[0].shape[0] == 1

    # Test with batch size 4
    out4 = session.run(None, {input_name: torch.randn(4, 3, 224, 224).numpy()})
    assert out4[0].shape[0] == 4


def test_quantized_model_smaller(trained_model, sample_input, tmp_path):
    """INT8 quantized model is smaller than the original ONNX model."""
    from src.worker.optimize import export_onnx, quantize_int8

    onnx_path = str(tmp_path / "model.onnx")
    quant_path = str(tmp_path / "model_int8.onnx")

    export_onnx(trained_model, sample_input, onnx_path)
    quantize_int8(onnx_path, quant_path)

    onnx_size = os.path.getsize(onnx_path)
    quant_size = os.path.getsize(quant_path)

    assert quant_size < onnx_size


def test_benchmark_produces_results(trained_model, sample_input, tmp_path):
    """Benchmark runs and produces latency numbers for all 3 variants."""
    from src.worker.optimize import benchmark_models, export_onnx, quantize_int8

    onnx_path = str(tmp_path / "model.onnx")
    quant_path = str(tmp_path / "model_int8.onnx")

    export_onnx(trained_model, sample_input, onnx_path)
    quantize_int8(onnx_path, quant_path)

    results = benchmark_models(trained_model, onnx_path, quant_path, sample_input, num_runs=5)

    assert "pytorch" in results
    assert "onnx" in results
    assert "quantized_int8" in results

    for variant in ["pytorch", "onnx", "quantized_int8"]:
        assert "latency_ms_p50" in results[variant]
        assert "latency_ms_p95" in results[variant]
        assert "latency_ms_p99" in results[variant]
        assert "model_size_mb" in results[variant]


def test_optimization_report(trained_model, sample_input, tmp_path):
    """Optimization report JSON contains all expected fields."""
    from src.worker.optimize import run_optimization_pipeline

    output_dir = str(tmp_path / "artifacts")
    run_optimization_pipeline(trained_model, sample_input, output_dir)

    report_path = os.path.join(output_dir, "optimization_report.json")
    assert os.path.exists(report_path)

    with open(report_path) as f:
        report = json.load(f)

    assert "benchmarks" in report
    assert "speedup_ratios" in report
    assert "size_reduction_ratios" in report
    assert "onnx_vs_pytorch" in report["speedup_ratios"]
    assert "int8_vs_pytorch" in report["speedup_ratios"]
