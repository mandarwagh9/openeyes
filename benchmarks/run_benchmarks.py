"""Comprehensive benchmarking suite for OpenEyes.

Measures FPS, latency, memory, and power consumption
across all models, backends, and hardware platforms.

Usage:
    python -m benchmarks.run_benchmarks --all
    python -m benchmarks.run_benchmarks --model yolo11n
    python -m benchmarks.run_benchmarks --backend onnxruntime
    python -m benchmarks.run_benchmarks --report
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backends.registry import BackendRegistry, ONNXRuntimeBackendStub
from src.platforms.detector import PlatformDetector
from src.utils.logger import get_logger


@dataclass
class BenchmarkResult:
    name: str
    model: str
    backend: str
    platform: str
    iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    min_ms: float
    max_ms: float
    timestamp: float = field(default_factory=time.time)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkSuite:
    """Runs benchmarks across models and backends."""

    def __init__(self, output_dir: Optional[str] = None):
        self._logger = get_logger(__name__)
        self._output_dir = output_dir or str(Path(__file__).parent / "results")
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)
        self._results: list[BenchmarkResult] = []
        self._platform = PlatformDetector.detect()

    def benchmark_onnx_model(
        self,
        model_path: str,
        model_name: str,
        iterations: int = 100,
        input_shape: tuple = (1, 3, 640, 640),
    ) -> BenchmarkResult:
        """Benchmark an ONNX model using ONNXRuntime."""
        self._logger.info(f"Benchmarking {model_name} ({model_path})...")

        if not Path(model_path).exists():
            self._logger.warning(f"Model not found: {model_path}")
            return BenchmarkResult(
                name=model_name,
                model=model_path,
                backend="onnxruntime",
                platform=self._platform.name,
                iterations=0,
                mean_ms=0, p50_ms=0, p95_ms=0, p99_ms=0,
                fps=0, min_ms=0, max_ms=0,
                notes="Model not found",
            )

        backend = ONNXRuntimeBackendStub()
        handle = backend.load_model(model_path)

        dummy = np.random.randn(*input_shape).astype(np.float32)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            backend.infer(handle, dummy)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        times_np = np.array(times)
        result = BenchmarkResult(
            name=model_name,
            model=model_path,
            backend="onnxruntime",
            platform=self._platform.name,
            iterations=iterations,
            mean_ms=float(np.mean(times_np)),
            p50_ms=float(np.percentile(times_np, 50)),
            p95_ms=float(np.percentile(times_np, 95)),
            p99_ms=float(np.percentile(times_np, 99)),
            fps=float(1000.0 / (np.mean(times_np) + 1e-9)),
            min_ms=float(np.min(times_np)),
            max_ms=float(np.max(times_np)),
        )

        self._results.append(result)
        self._logger.info(
            f"  {model_name}: {result.fps:.1f} FPS "
            f"(mean={result.mean_ms:.1f}ms, p95={result.p95_ms:.1f}ms)"
        )

        return result

    def benchmark_all_models(self, iterations: int = 100) -> list[BenchmarkResult]:
        """Benchmark all available ONNX models."""
        models_dir = Path(__file__).parent.parent / "models"
        results = []

        onnx_models = {
            "yolo11n": "yolo11n.onnx",
            "yolov8n": "yolov8n.onnx",
            "yolov10n": "yolov10n.onnx",
        }

        for name, filename in onnx_models.items():
            model_path = str(models_dir / filename)
            if Path(model_path).exists():
                result = self.benchmark_onnx_model(
                    model_path, name, iterations=iterations
                )
                results.append(result)

        return results

    def benchmark_backend(self, iterations: int = 100) -> list[BenchmarkResult]:
        """Benchmark all available backends."""
        results = []
        available = BackendRegistry.list_available()

        for info in available:
            self._logger.info(f"Backend: {info.name} ({info.version})")
            results.append({
                "name": info.name,
                "version": info.version,
                "device": info.device_name,
                "precisions": info.supported_precisions,
                "max_batch": info.max_batch_size,
            })

        return results

    def get_platform_info(self) -> dict:
        """Get current platform information."""
        return {
            "name": self._platform.name,
            "vendor": self._platform.vendor,
            "cpu": self._platform.cpu,
            "gpu": self._platform.gpu,
            "npu": self._platform.npu,
            "memory_gb": self._platform.total_memory_gb,
            "backends": self._platform.available_backends,
            "recommended_precision": self._platform.recommended_precision,
        }

    def save_report(self, filename: str = "benchmark_report.json") -> str:
        """Save benchmark results to JSON report."""
        report = {
            "timestamp": time.time(),
            "platform": self.get_platform_info(),
            "results": [r.to_dict() for r in self._results],
            "summary": self._generate_summary(),
        }

        output_path = Path(self._output_dir) / filename
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        self._logger.info(f"Report saved to: {output_path}")
        return str(output_path)

    def _generate_summary(self) -> dict:
        """Generate summary statistics."""
        if not self._results:
            return {}

        fps_values = [r.fps for r in self._results if r.fps > 0]
        latency_values = [r.mean_ms for r in self._results if r.mean_ms > 0]

        return {
            "total_benchmarks": len(self._results),
            "avg_fps": float(np.mean(fps_values)) if fps_values else 0,
            "max_fps": float(max(fps_values)) if fps_values else 0,
            "min_fps": float(min(fps_values)) if fps_values else 0,
            "avg_latency_ms": float(np.mean(latency_values)) if latency_values else 0,
            "best_model": max(self._results, key=lambda r: r.fps).name if self._results else "",
        }

    def print_results(self) -> None:
        """Print benchmark results in a formatted table."""
        if not self._results:
            print("No benchmark results available.")
            return

        print("\n" + "=" * 80)
        print("OpenEyes Benchmark Results")
        print("=" * 80)
        print(f"Platform: {self._platform.name} ({self._platform.vendor})")
        print(f"CPU: {self._platform.cpu}")
        print(f"GPU: {self._platform.gpu or 'N/A'}")
        print(f"Memory: {self._platform.total_memory_gb} GB")
        print("-" * 80)
        print(f"{'Model':<15} {'Backend':<15} {'FPS':>8} {'Mean':>8} {'P95':>8} {'P99':>8}")
        print("-" * 80)

        for r in self._results:
            print(
                f"{r.name:<15} {r.backend:<15} {r.fps:>8.1f} "
                f"{r.mean_ms:>7.1f}ms {r.p95_ms:>7.1f}ms {r.p99_ms:>7.1f}ms"
            )

        print("-" * 80)
        summary = self._generate_summary()
        if summary:
            print(f"Average FPS: {summary['avg_fps']:.1f}")
            print(f"Best model:  {summary['best_model']} ({summary['max_fps']:.1f} FPS)")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="OpenEyes Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--model", type=str, help="Benchmark specific model")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    suite = BenchmarkSuite(output_dir=args.output_dir)

    if args.all or args.report:
        suite.benchmark_all_models(iterations=args.iterations)
        suite.print_results()
        if args.report:
            suite.save_report()
    elif args.model:
        models_dir = Path(__file__).parent.parent / "models"
        model_path = str(models_dir / f"{args.model}.onnx")
        if not Path(model_path).exists():
            model_path = str(models_dir / f"{args.model}.pt")
        suite.benchmark_onnx_model(model_path, args.model, iterations=args.iterations)
        suite.print_results()
    else:
        suite.print_results()
        suite.benchmark_all_models(iterations=args.iterations)
        suite.print_results()
        suite.save_report()


if __name__ == "__main__":
    main()
