#!/usr/bin/env python3
"""Batch performance harness for the Week 03 MNN MobileNetV2 baseline.

This script wraps the existing `mnn_mobilenetv2` executable and turns it into
repeatable batch performance evidence:

- runs a small local image manifest
- captures top-1 prediction and timing statistics
- optionally compares CPU and Metal requests
- prints a Markdown table that can be copied into reports
- writes a Markdown report to `reports/week3_mnn_inference_baseline.md`

The goal is not to reimplement inference in Python. The goal is to make the
existing C++ baseline easy to measure, compare, and archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "experiments" / "week03-mnn-inference-engine"
BUILD_DIR = PROJECT_DIR / "build"
BIN_PATH = BUILD_DIR / "mnn_mobilenetv2"
MODEL_PATH = REPO_ROOT / "assets" / "models" / "mobilenetv2.mnn"
LABELS_PATH = REPO_ROOT / "assets" / "models" / "imagenet_classes.txt"
MANIFEST_PATH = REPO_ROOT / "assets" / "models" / "eval_manifest.csv"
IMAGE_DIR = REPO_ROOT / "assets" / "models"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "week3_mnn_inference_baseline.md"
DEFAULT_JSON_REPORT_PATH = REPO_ROOT / "reports" / "week3_mnn_inference_baseline.json"


TOP1_RE = re.compile(
    r'^\s*class=(?P<idx>\d+)\s+prob=(?P<prob>[\d.]+)\s+label="(?P<label>.*)"\s*$'
)
MEDIAN_RE = re.compile(r"^Median:\s+(?P<value>[\d.]+)\s+ms\s*$")
MEAN_RE = re.compile(r"^Mean:\s+(?P<value>[\d.]+)\s+ms\s*$")
P95_RE = re.compile(r"^P95:\s+(?P<value>[\d.]+)\s+ms\s*$")
BACKEND_RE = re.compile(r"^Requested backend:\s+(?P<backend>.+)$")


@dataclass
class SampleResult:
    image_name: str
    expected_idx: int
    expected_label: str
    backend_key: str
    top1_idx: int
    top1_label: str
    top1_prob: float
    requested_backend: str
    load_ms: float | None
    mean_ms: float
    median_ms: float
    p95_ms: float


def display_backend(backend: str) -> str:
    normalized = backend.strip().lower()
    if "cpu" in normalized:
        return "CPU"
    if "metal" in normalized:
        return "Metal"
    if "vulkan" in normalized:
        return "Vulkan"
    return backend.strip().title()


def display_prob_percent(probability: float) -> str:
    value = probability * 100.0 if probability <= 1.0 else probability
    return f"{value:.3f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["cpu", "metal", "both"],
        default="cpu",
        help="Which backend request(s) to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs per invocation.")
    parser.add_argument("--runs", type=int, default=100, help="Timed runs per invocation.")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the Markdown report.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=DEFAULT_JSON_REPORT_PATH,
        help="Where to write the JSON report.",
    )
    return parser.parse_args()


def load_labels() -> list[str]:
    labels = LABELS_PATH.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in labels if line.strip()]


def load_manifest() -> tuple[list[tuple[str, int]], list[str]]:
    rows: list[tuple[str, int]] = []
    missing: list[str] = []
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image"].strip()
            label = int(row["label"])
            image_path = IMAGE_DIR / image_name
            if not image_path.exists():
                missing.append(f"{image_name} -> {image_path}")
                continue
            rows.append((image_name, label))
    return rows, missing


def run_binary(image_path: Path, backend: str, warmup: int, runs: int) -> SampleResult:
    result = subprocess.run(
        [
            str(BIN_PATH),
            str(MODEL_PATH),
            str(image_path),
            backend,
            str(LABELS_PATH),
            str(warmup),
            str(runs),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}")

    top1_idx = None
    top1_label = None
    top1_prob = None
    requested_backend = None
    mean_ms = None
    median_ms = None
    p95_ms = None

    for line in result.stdout.splitlines():
        if top1_idx is None:
            match = TOP1_RE.match(line)
            if match:
                top1_idx = int(match.group("idx"))
                top1_label = match.group("label")
                top1_prob = float(match.group("prob"))
                continue

        if requested_backend is None:
            match = BACKEND_RE.match(line)
            if match:
                requested_backend = match.group("backend")
                continue

        if mean_ms is None:
            match = MEAN_RE.match(line)
            if match:
                mean_ms = float(match.group("value"))
                continue

        if median_ms is None:
            match = MEDIAN_RE.match(line)
            if match:
                median_ms = float(match.group("value"))
                continue

        if p95_ms is None:
            match = P95_RE.match(line)
            if match:
                p95_ms = float(match.group("value"))
                continue

    if top1_idx is None or top1_label is None or top1_prob is None:
        raise RuntimeError(f"failed to parse top-1 result for {image_path.name}")
    if requested_backend is None or mean_ms is None or median_ms is None or p95_ms is None:
        raise RuntimeError(f"failed to parse timing result for {image_path.name}")

    return SampleResult(
        image_name=image_path.name,
        expected_idx=-1,
        expected_label="",
        backend_key=backend,
        top1_idx=top1_idx,
        top1_label=top1_label,
        top1_prob=top1_prob,
        requested_backend=requested_backend,
        load_ms=None,
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
    )


def render_report(
    report_path: Path,
    rows: list[SampleResult],
    missing_entries: list[str],
    warmup: int,
    runs: int,
    requested_backends: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def mean_of(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def accuracy(samples: list[SampleResult]) -> float:
        if not samples:
            return 0.0
        matches = sum(1 for sample in samples if sample.top1_label == sample.expected_label)
        return matches / len(samples) * 100.0

    def backend_summary(label: str) -> tuple[str, str, str, str, str]:
        samples = grouped.get(label, [])
        if not samples:
            return ("—", "—", "—", "—", "—")
        return (
            "—",
            f"{mean_of([sample.median_ms for sample in samples]):.3f}",
            f"{mean_of([sample.p95_ms for sample in samples]):.3f}",
            display_prob_percent(mean_of([sample.top1_prob for sample in samples])),
            f"{accuracy(samples):.2f}%",
        )

    grouped: dict[str, list[SampleResult]] = {}
    for row in rows:
        grouped.setdefault(row.backend_key, []).append(row)

    lines: list[str] = []
    lines.append("# Week 3 macOS Inference Baseline")
    lines.append("")
    lines.append("## 1. Environment")
    lines.append("- 设备: Mac M1 Pro")
    lines.append("- Platform: macOS")
    lines.append("- OS: macOS")
    lines.append("- CPU: Apple M1 Pro")
    lines.append("- Memory: —")
    lines.append("- MNN commit / version: —")
    lines.append("- NCNN commit / version: —")
    lines.append("- OpenCV version: —")
    lines.append("- Runtime: MNN")
    lines.append(f"- 测试轮数: {runs}")
    lines.append(f"- 预热轮数: {warmup}")
    lines.append(f"- 请求后端: {', '.join(display_backend(backend) for backend in requested_backends)}")
    lines.append("")
    lines.append("## 2. Model")
    lines.append("- Model: MobileNetV2")
    lines.append("")
    lines.append("- Source: `assets/models/mobilenetv2.onnx`")
    lines.append("- Input: `1x3x224x224`")
    lines.append("- Output: `1x1000`")
    lines.append("- Preprocess:")
    lines.append("  - resize: `224x224`")
    lines.append("  - color: `BGR -> RGB`")
    lines.append("  - mean: `[123.675, 116.28, 103.53]`")
    lines.append("  - norm: `[1/58.395, 1/57.12, 1/57.375]`")
    lines.append("")
    lines.append("## 3. Latency")
    lines.append("")
    lines.append("| Framework | Backend | Device | Load ms | Warmup | Median ms | P95 ms | Top1 | Notes |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    load_ms, median_ms, p95_ms, prob, acc = backend_summary("cpu")
    lines.append(
        f"| MNN | {display_backend('cpu')} | Mac M1 Pro | {load_ms} | {warmup} | {median_ms} | {p95_ms} | {acc} | top1 prob avg {prob} |"
    )
    load_ms, median_ms, p95_ms, prob, acc = backend_summary("metal")
    lines.append(
        f"| MNN | {display_backend('metal')} | Mac M1 Pro | {load_ms} | {warmup} | {median_ms} | {p95_ms} | {acc} | requested backend {display_backend('metal') if 'metal' in requested_backends else '—'} |"
    )
    lines.append("| NCNN | CPU | Mac M1 Pro | — | — | — | — | — | not measured in MNN script |")
    lines.append("| NCNN | Vulkan | Mac M1 Pro | — | — | — | — | — | not measured in MNN script |")
    lines.append("")
    lines.append("## 4. Result Check")
    lines.append("")
    lines.append("- MNN top-5:")
    lines.append("- NCNN top-5:")
    lines.append("- Difference:")
    lines.append("")
    lines.append("## 5. Engineering Notes")
    lines.append("")
    lines.append("- Model conversion issues:")
    lines.append("- Input/output name issues:")
    lines.append("- Preprocess issues:")
    lines.append("- Backend difference:")
    lines.append("- What can be reused in Week 4:")
    lines.append("- What can be reused in Week 5:")
    lines.append("")
    lines.append("## Appendix: Sample Results")
    lines.append("")
    lines.append("| Image | Backend | Expected idx | Expected label | Actual idx | Actual label | Prob | Load ms | Mean ms | Median ms | P95 ms | Match | Note |")
    lines.append("| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for sample in rows:
        load_str = "—" if sample.load_ms is None else f"{sample.load_ms:.3f}"
        match = sample.top1_label == sample.expected_label
        note = ""
        lines.append(
            f"| {sample.image_name} | {display_backend(sample.requested_backend)} | {sample.expected_idx} | {sample.expected_label} | "
            f"{sample.top1_idx} | {sample.top1_label} | {display_prob_percent(sample.top1_prob)} | {load_str} | "
            f"{sample.mean_ms:.3f} | {sample.median_ms:.3f} | {sample.p95_ms:.3f} | {'YES' if match else 'NO'} | {note} |"
        )

    if missing_entries:
        lines.append("")
        lines.append("## Missing Images")
        lines.append("")
        for item in missing_entries:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This report fixes the week03 MNN baseline and is not a standalone accuracy benchmark.")
    lines.append("- Top-1 matching against the local manifest is the correctness criterion for this script.")
    lines.append("- If `Metal` falls back to CPU at runtime, the requested backend is still recorded as `Metal`.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_report_payload(
    rows: list[SampleResult],
    missing_entries: list[str],
    warmup: int,
    runs: int,
    requested_backends: list[str],
) -> dict[str, object]:
    def mean_of(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    grouped: dict[str, list[SampleResult]] = {}
    for row in rows:
        grouped.setdefault(row.backend_key, []).append(row)

    def summary_row(backend: str, framework: str) -> dict[str, object]:
        samples = grouped.get(backend, [])
        if not samples:
            return {
                "framework": framework,
                "backend": display_backend(backend),
                "device": "Mac M1 Pro",
                "loadMs": 0.0,
                "warmupRuns": warmup,
                "benchmarkRuns": runs,
                "meanMs": 0.0,
                "medianMs": 0.0,
                "p95Ms": 0.0,
                "top1Idx": -1,
                "top1Label": "",
                "top1Prob": 0.0,
                "notes": "not measured",
            }

        first = samples[0]
        return {
            "framework": framework,
            "backend": display_backend(backend),
            "device": "Mac M1 Pro",
            "loadMs": mean_of([sample.load_ms or 0.0 for sample in samples]),
            "warmupRuns": warmup,
            "benchmarkRuns": runs,
            "meanMs": mean_of([sample.mean_ms for sample in samples]),
                "medianMs": mean_of([sample.median_ms for sample in samples]),
                "p95Ms": mean_of([sample.p95_ms for sample in samples]),
                "top1Idx": first.top1_idx,
                "top1Label": first.top1_label,
                "top1Prob": first.top1_prob / 100.0 if first.top1_prob > 1.0 else first.top1_prob,
                "notes": f"top1 prob avg {display_prob_percent(mean_of([sample.top1_prob for sample in samples]))}",
            }

    return {
        "title": "Week 3 macOS Inference Baseline",
        "environment": [
            {"key": "设备", "value": "Mac M1 Pro"},
            {"key": "Platform", "value": "macOS"},
            {"key": "OS", "value": "macOS"},
            {"key": "CPU", "value": "Apple M1 Pro"},
            {"key": "Memory", "value": "—"},
            {"key": "MNN commit / version", "value": "—"},
            {"key": "NCNN commit / version", "value": "—"},
            {"key": "OpenCV version", "value": "—"},
            {"key": "Runtime", "value": "MNN"},
            {"key": "测试轮数", "value": str(runs)},
            {"key": "预热轮数", "value": str(warmup)},
            {"key": "请求后端", "value": ", ".join(display_backend(backend) for backend in requested_backends)},
        ],
        "model": [
            {"key": "Model", "value": "MobileNetV2"},
            {"key": "Source", "value": "assets/models/mobilenetv2.onnx"},
            {"key": "Input", "value": "1x3x224x224"},
            {"key": "Output", "value": "1x1000"},
            {"key": "Preprocess", "value": "ImageNet resize / crop / normalize"},
        ],
        "latencyRows": [
            summary_row("cpu", "MNN"),
            summary_row("metal", "MNN"),
            {
                "framework": "NCNN",
                "backend": "CPU",
                "device": "Mac M1 Pro",
                "loadMs": 0.0,
                "warmupRuns": 0,
                "benchmarkRuns": 0,
                "meanMs": 0.0,
                "medianMs": 0.0,
                "p95Ms": 0.0,
                "top1Idx": -1,
                "top1Label": "",
                "top1Prob": 0.0,
                "notes": "not measured in MNN script",
            },
            {
                "framework": "NCNN",
                "backend": "Vulkan",
                "device": "Mac M1 Pro",
                "loadMs": 0.0,
                "warmupRuns": 0,
                "benchmarkRuns": 0,
                "meanMs": 0.0,
                "medianMs": 0.0,
                "p95Ms": 0.0,
                "top1Idx": -1,
                "top1Label": "",
                "top1Prob": 0.0,
                "notes": "not measured in MNN script",
            },
        ],
        "resultChecks": [
            "MNN top-5:",
            "NCNN top-5:",
            "Difference:",
        ],
        "engineeringNotes": [
            "Model conversion issues:",
            "Input/output name issues:",
            "Preprocess issues:",
            "Backend difference:",
            "What can be reused in Week 4:",
            "What can be reused in Week 5:",
        ],
        "sampleRows": [
            {
                "imageName": result.image_name,
                "backend": display_backend(result.backend_key),
                "expectedIdx": result.expected_idx,
                "expectedLabel": result.expected_label,
                "top1Idx": result.top1_idx,
                "top1Label": result.top1_label,
                "top1Prob": result.top1_prob / 100.0 if result.top1_prob > 1.0 else result.top1_prob,
                "loadMs": result.load_ms,
                "meanMs": result.mean_ms,
                "medianMs": result.median_ms,
                "p95Ms": result.p95_ms,
                "note": "",
            }
            for result in rows
        ],
        "appendixNotes": [
            "This report fixes the week03 MNN baseline and is not a standalone accuracy benchmark.",
            "Top-1 matching against the local manifest is the correctness criterion for this script.",
            "If `Metal` falls back to CPU at runtime, the requested backend is still recorded as `Metal`.",
        ],
        "missingImages": missing_entries,
    }


def write_json_report(report_path: Path, rows: list[SampleResult], missing_entries: list[str], warmup: int, runs: int, requested_backends: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload(rows, missing_entries, warmup, runs, requested_backends)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if not BIN_PATH.exists():
        print(f"missing binary: {BIN_PATH}", file=sys.stderr)
        return 2
    if not MODEL_PATH.exists():
        print(f"missing model: {MODEL_PATH}", file=sys.stderr)
        return 2
    if not LABELS_PATH.exists():
        print(f"missing labels file: {LABELS_PATH}", file=sys.stderr)
        return 2
    if not MANIFEST_PATH.exists():
        print(f"missing manifest: {MANIFEST_PATH}", file=sys.stderr)
        return 2

    labels = load_labels()
    manifest_rows, missing_entries = load_manifest()
    if not manifest_rows:
        print("no valid image rows found in manifest", file=sys.stderr)
        return 2

    requested_backends = ["cpu", "metal"] if args.backend == "both" else [args.backend]

    print("| image | backend | expected idx | expected label | actual idx | actual label | prob | load ms | mean ms | median ms | p95 ms | match | note |")
    print("| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")

    sample_results: list[SampleResult] = []
    for image_name, expected_idx in manifest_rows:
        image_path = IMAGE_DIR / image_name
        expected_label = labels[expected_idx] if 0 <= expected_idx < len(labels) else ""

        for backend in requested_backends:
            metrics = run_binary(image_path, backend, args.warmup, args.runs)
            metrics.expected_idx = expected_idx
            metrics.expected_label = expected_label
            metrics.backend_key = backend
            match = metrics.top1_label == metrics.expected_label
            note = ""
            print(
                f"| {image_name} | {display_backend(metrics.requested_backend)} | {metrics.expected_idx} | {metrics.expected_label} | "
                f"{metrics.top1_idx} | {metrics.top1_label} | {display_prob_percent(metrics.top1_prob)} | "
                f"{'—' if metrics.load_ms is None else f'{metrics.load_ms:.3f}'} | {metrics.mean_ms:.3f} | "
                f"{metrics.median_ms:.3f} | {metrics.p95_ms:.3f} | {'YES' if match else 'NO'} | {note} |"
            )
            sample_results.append(metrics)

    render_report(args.report, sample_results, missing_entries, args.warmup, args.runs, requested_backends)
    write_json_report(args.json_report, sample_results, missing_entries, args.warmup, args.runs, requested_backends)
    print(f"\nwritten report: {args.report}")
    print(f"written json report: {args.json_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
