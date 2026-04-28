#!/usr/bin/env python3
"""Batch performance and regression harness for the Week 03 MobileNetV2 NCNN baseline.

This script measures a small local image set and records the observed
predictions/timings against the manifest labels.

Important:
- The manifest label is the correctness target.
- The historical baseline mapping is kept only as a reference for notes and
  drift analysis.
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
PROJECT_DIR = REPO_ROOT / "experiments" / "week03-ncnn-conversion"
BUILD_DIR = PROJECT_DIR / "build"
BIN_PATH = BUILD_DIR / "mobilenetv2_ncnn"
MODEL_PATH = PROJECT_DIR / "outputs" / "mobilenetv2.param"
WEIGHTS_PATH = PROJECT_DIR / "outputs" / "mobilenetv2.bin"
LABELS_PATH = REPO_ROOT / "assets" / "models" / "imagenet_classes.txt"
MANIFEST_PATH = REPO_ROOT / "assets" / "models" / "eval_manifest.csv"
IMAGE_DIR = REPO_ROOT / "assets" / "models"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "week3_ncnn_inference_baseline.md"
DEFAULT_JSON_REPORT_PATH = REPO_ROOT / "reports" / "week3_ncnn_inference_baseline.json"


# Historical reference predictions recorded from the ONNX Runtime workflow.
# These are used for drift notes, not for the primary expected-label field.
BASELINE_TOP1 = {
    "test_dog.jpg": ("Samoyed", 258),
    "dog2.jpg": ("golden retriever", 207),
    "tabby_cat.jpg": ("tabby", 281),
    "tiger_cat.jpg": ("jaguar", 291),
    "goldfish.jpg": ("goldfish", 1),
    "macaw.jpg": ("macaw", 88),
    "sports_car.jpg": ("sports car", 817),
}


TOP1_RE = re.compile(
    r"\[Result\]\s+top1_idx=(?P<idx>\d+)\s+top1_label=(?P<label>.+?)\s+top1_prob=(?P<prob>[\d.]+)%"
)
LOAD_RE = re.compile(r"^\[Load\]\s+model_load_ms=(?P<value>[\d.]+)\s*$", re.MULTILINE)
BENCH_RE = re.compile(
    r"^\[Benchmark\]\s+runs=(?P<runs>\d+)\s+mean_ms=(?P<mean>[\d.]+)\s+median_ms=(?P<median>[\d.]+)\s+p95_ms=(?P<p95>[\d.]+)\s*$",
    re.MULTILINE,
)


@dataclass
class SampleResult:
    image_name: str
    expected_idx: int
    expected_label: str
    backend_key: str
    top1_idx: int
    top1_label: str
    top1_prob: float
    load_ms: float
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
    parser.add_argument("--backend", choices=["cpu", "vulkan", "both"], default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT_PATH)
    return parser.parse_args()


def run_one(image_path: Path, backend: str, warmup: int, runs: int) -> dict[str, object]:
    result = subprocess.run(
        [
            str(BIN_PATH),
            "--param",
            str(MODEL_PATH),
            "--bin",
            str(WEIGHTS_PATH),
            "--image",
            str(image_path),
            "--labels",
            str(LABELS_PATH),
            "--backend",
            backend,
            "--topk",
            "5",
            "--warmup",
            str(warmup),
            "--runs",
            str(runs),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}")

    top1_match = TOP1_RE.search(result.stdout)
    load_match = LOAD_RE.search(result.stdout)
    bench_match = BENCH_RE.search(result.stdout)
    if not top1_match:
        raise RuntimeError(f"failed to parse top1 line for {image_path.name}")
    if not load_match or not bench_match:
        raise RuntimeError(f"failed to parse timing lines for {image_path.name}")

    return {
        "idx": int(top1_match.group("idx")),
        "label": top1_match.group("label"),
        "prob": float(top1_match.group("prob")),
        "load_ms": float(load_match.group("value")),
        "mean_ms": float(bench_match.group("mean")),
        "median_ms": float(bench_match.group("median")),
        "p95_ms": float(bench_match.group("p95")),
        "backend": backend,
    }


def load_manifest() -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image"].strip()
            label = int(row["label"])
            image_path = IMAGE_DIR / image_name
            if not image_path.exists():
                continue
            rows.append((image_name, label))
    return rows


def load_labels() -> list[str]:
    return [line.strip() for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_report(
    report_path: Path,
    results: list[SampleResult],
    missing_entries: list[str],
    warmup: int,
    runs: int,
    requested_backends: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def mean_of(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def summarize(samples: list[SampleResult]) -> tuple[float, float, float, float, float]:
        if not samples:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        matched = sum(1 for sample in samples if sample.top1_label == sample.expected_label)
        accuracy = matched / len(samples) * 100.0
        mean_load = mean_of([sample.load_ms for sample in samples])
        mean_median = mean_of([sample.median_ms for sample in samples])
        mean_p95 = mean_of([sample.p95_ms for sample in samples])
        mean_prob = mean_of([sample.top1_prob for sample in samples])
        return mean_load, mean_median, mean_p95, mean_prob, accuracy

    grouped: dict[str, list[SampleResult]] = {}
    for row in results:
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
    lines.append("- Xcode version: —")
    lines.append("- Runtime: NCNN")
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
    lines.append("| MNN | CPU | Mac M1 Pro | — | — | — | — | — | not measured in NCNN script |")
    lines.append("| MNN | Metal | Mac M1 Pro | — | — | — | — | — | not measured in NCNN script |")

    for backend in ["cpu", "vulkan"]:
        samples = grouped.get(backend, [])
        if samples:
            mean_load, mean_median, mean_p95, mean_prob, accuracy = summarize(samples)
            lines.append(
                f"| NCNN | {display_backend(backend)} | Mac M1 Pro | {mean_load:.3f} | {warmup} | {mean_median:.3f} | {mean_p95:.3f} | {accuracy:.2f}% | top1 prob avg {display_prob_percent(mean_prob)} |"
            )
        else:
            lines.append(f"| NCNN | {display_backend(backend)} | Mac M1 Pro | — | — | — | — | — | not measured in NCNN script |")

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
    lines.append("| Image | Backend | Expected label | Top-1 prediction | Top-1 prob | Load ms | Mean ms | Median ms | P95 ms |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        lines.append(
            f"| {result.image_name} | {display_backend(result.backend_key)} | {result.expected_label} | {result.top1_label} | "
            f"{display_prob_percent(result.top1_prob)} | {result.load_ms:.3f} | {result.mean_ms:.3f} | {result.median_ms:.3f} | {result.p95_ms:.3f} |"
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
    lines.append("- This report fixes the week03 NCNN baseline and is not a standalone accuracy benchmark.")
    lines.append("- Top-1 matching against the local manifest is the correctness criterion for this script.")
    lines.append("- NCNN on Apple platforms uses the Vulkan path; Metal is not a native NCNN backend.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_report_payload(
    results: list[SampleResult],
    missing_entries: list[str],
    warmup: int,
    runs: int,
    requested_backends: list[str],
) -> dict[str, object]:
    def mean_of(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def summarize(samples: list[SampleResult]) -> dict[str, object]:
        if not samples:
            return {
                "loadMs": 0.0,
                "warmupRuns": 0,
                "benchmarkRuns": 0,
                "meanMs": 0.0,
                "medianMs": 0.0,
                "p95Ms": 0.0,
                "top1Idx": -1,
                "top1Label": "",
                "top1Prob": 0.0,
                "notes": "not measured",
            }

        matched = sum(1 for sample in samples if sample.top1_label == sample.expected_label)
        mean_prob = mean_of([sample.top1_prob for sample in samples])
        return {
            "loadMs": mean_of([sample.load_ms for sample in samples]),
            "warmupRuns": warmup,
            "benchmarkRuns": runs,
            "meanMs": mean_of([sample.mean_ms for sample in samples]),
            "medianMs": mean_of([sample.median_ms for sample in samples]),
            "p95Ms": mean_of([sample.p95_ms for sample in samples]),
            "top1Idx": samples[0].top1_idx,
            "top1Label": samples[0].top1_label,
            "top1Prob": mean_prob / 100.0 if mean_prob > 1.0 else mean_prob,
            "notes": f"top1 prob avg {display_prob_percent(mean_prob)}; accuracy {matched / len(samples) * 100.0:.2f}%",
        }

    grouped: dict[str, list[SampleResult]] = {}
    for row in results:
        grouped.setdefault(row.backend_key, []).append(row)
    cpu_summary = summarize(grouped.get("cpu", []))
    vulkan_summary = summarize(grouped.get("vulkan", []))

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
            {"key": "Xcode version", "value": "—"},
            {"key": "Runtime", "value": "NCNN"},
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
            {
                "framework": "MNN",
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
                "notes": "not measured in NCNN script",
            },
            {
                "framework": "MNN",
                "backend": "Metal",
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
                "notes": "not measured in NCNN script",
            },
            {
                "framework": "NCNN",
                "backend": "CPU",
                "device": "Mac M1 Pro",
                "loadMs": cpu_summary.get("loadMs", 0.0),
                "warmupRuns": warmup if grouped.get("cpu") else 0,
                "benchmarkRuns": runs if grouped.get("cpu") else 0,
                "meanMs": cpu_summary.get("meanMs", 0.0),
                "medianMs": cpu_summary.get("medianMs", 0.0),
                "p95Ms": cpu_summary.get("p95Ms", 0.0),
                "top1Idx": cpu_summary.get("top1Idx", -1),
                "top1Label": cpu_summary.get("top1Label", ""),
                "top1Prob": cpu_summary.get("top1Prob", 0.0),
                "notes": cpu_summary.get("notes", "not measured"),
            },
            {
                "framework": "NCNN",
                "backend": "Vulkan",
                "device": "Mac M1 Pro",
                "loadMs": vulkan_summary.get("loadMs", 0.0),
                "warmupRuns": warmup if grouped.get("vulkan") else 0,
                "benchmarkRuns": runs if grouped.get("vulkan") else 0,
                "meanMs": vulkan_summary.get("meanMs", 0.0),
                "medianMs": vulkan_summary.get("medianMs", 0.0),
                "p95Ms": vulkan_summary.get("p95Ms", 0.0),
                "top1Idx": vulkan_summary.get("top1Idx", -1),
                "top1Label": vulkan_summary.get("top1Label", ""),
                "top1Prob": vulkan_summary.get("top1Prob", 0.0),
                "notes": vulkan_summary.get("notes", "not measured"),
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
            for result in results
        ],
        "appendixNotes": [
            "This report fixes the week03 NCNN baseline and is not a standalone accuracy benchmark.",
            "Top-1 matching against the local manifest is the correctness criterion for this script.",
            "NCNN on Apple platforms uses the Vulkan path; Metal is not a native NCNN backend.",
        ],
        "missingImages": missing_entries,
    }


def write_json_report(
    report_path: Path,
    results: list[SampleResult],
    missing_entries: list[str],
    warmup: int,
    runs: int,
    requested_backends: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload(results, missing_entries, warmup, runs, requested_backends)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if not BIN_PATH.exists():
        print(f"missing binary: {BIN_PATH}", file=sys.stderr)
        return 2
    if not MODEL_PATH.exists():
        print(f"missing model param: {MODEL_PATH}", file=sys.stderr)
        return 2
    if not WEIGHTS_PATH.exists():
        print(f"missing model weights: {WEIGHTS_PATH}", file=sys.stderr)
        return 2
    if not LABELS_PATH.exists():
        print(f"missing labels file: {LABELS_PATH}", file=sys.stderr)
        return 2
    if not MANIFEST_PATH.exists():
        print(f"missing manifest: {MANIFEST_PATH}", file=sys.stderr)
        return 2

    labels = load_labels()
    rows = load_manifest()
    if not rows:
        print("no valid images found in manifest", file=sys.stderr)
        return 2

    requested_backends = ["cpu", "vulkan"] if args.backend == "both" else [args.backend]

    print("| image | backend | expected idx | expected label | actual idx | actual label | prob | load ms | mean ms | median ms | p95 ms | match | note |")
    print("| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")

    failures: list[str] = []
    matched = 0
    sample_results: list[SampleResult] = []

    for backend in requested_backends:
        for image_name, expected_idx in rows:
            image_path = IMAGE_DIR / image_name
            expected_label = labels[expected_idx] if 0 <= expected_idx < len(labels) else "<unknown>"
            baseline_reference = BASELINE_TOP1.get(image_name)
            result = run_one(image_path, backend, args.warmup, args.runs)
            actual_idx = int(result["idx"])
            actual_label = str(result["label"])
            prob = float(result["prob"])
            load_ms = float(result["load_ms"])
            mean_ms = float(result["mean_ms"])
            median_ms = float(result["median_ms"])
            p95_ms = float(result["p95_ms"])
            ok = actual_label == expected_label
            note = ""
            if baseline_reference is not None:
                baseline_label, baseline_idx = baseline_reference
                if baseline_label != expected_label or baseline_idx != expected_idx:
                    note = f"baseline ref: {baseline_idx}/{baseline_label}"
            matched += int(ok)
            if not ok:
                failures.append(
                    f"{image_name} [{display_backend(backend)}]: expected {expected_idx}/{expected_label}, got {actual_idx}/{actual_label}"
                )
            print(
                f"| {image_name} | {display_backend(backend)} | {expected_idx} | {expected_label} | "
                f"{actual_idx} | {actual_label} | {display_prob_percent(prob)} | {load_ms:.3f} | {mean_ms:.3f} | "
                f"{median_ms:.3f} | {p95_ms:.3f} | {'YES' if ok else 'NO'} | {note} |"
            )

            sample_results.append(
                SampleResult(
                    image_name=image_name,
                    expected_idx=expected_idx,
                    expected_label=expected_label,
                    backend_key=backend,
                    top1_idx=actual_idx,
                    top1_label=actual_label,
                    top1_prob=prob,
                    load_ms=load_ms,
                    mean_ms=mean_ms,
                    median_ms=median_ms,
                    p95_ms=p95_ms,
                )
            )

    print(f"\nmatched {matched}/{len(rows) * len(requested_backends)} manifest labels")

    render_report(args.report, sample_results, [], args.warmup, args.runs, requested_backends)
    print(f"written report: {args.report}")
    write_json_report(args.json_report, sample_results, [], args.warmup, args.runs, requested_backends)
    print(f"written json report: {args.json_report}")

    if failures:
        print("\nregressions detected:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
