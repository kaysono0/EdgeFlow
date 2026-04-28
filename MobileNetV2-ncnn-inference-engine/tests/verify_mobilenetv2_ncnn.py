#!/usr/bin/env python3
"""Smoke test for the Week 03 MobileNetV2 NCNN demo.

This test runs the compiled C++ binary on a fixed ImageNet sample and checks:
- the program exits successfully
- the reported top-1 class is Samoyed
- the reported top-1 class id is 258

It is intentionally strict enough to catch preprocessing / model-loading bugs,
but it stays narrow so the demo remains easy to iterate on.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "experiments" / "week03-ncnn-conversion"
BUILD_DIR = PROJECT_DIR / "build"
BIN_PATH = BUILD_DIR / "mobilenetv2_ncnn"
MODEL_PATH = PROJECT_DIR / "outputs" / "mobilenetv2.param"
WEIGHTS_PATH = PROJECT_DIR / "outputs" / "mobilenetv2.bin"
IMAGE_PATH = REPO_ROOT / "assets" / "models" / "test_dog.jpg"
LABELS_PATH = REPO_ROOT / "assets" / "models" / "imagenet_classes.txt"


def main() -> int:
    if not BIN_PATH.exists():
        print(f"missing binary: {BIN_PATH}", file=sys.stderr)
        return 2
    if not MODEL_PATH.exists():
        print(f"missing model param: {MODEL_PATH}", file=sys.stderr)
        return 2
    if not WEIGHTS_PATH.exists():
        print(f"missing model weights: {WEIGHTS_PATH}", file=sys.stderr)
        return 2
    if not IMAGE_PATH.exists():
        print(f"missing test image: {IMAGE_PATH}", file=sys.stderr)
        return 2
    if not LABELS_PATH.exists():
        print(f"missing labels file: {LABELS_PATH}", file=sys.stderr)
        return 2

    result = subprocess.run(
        [
            str(BIN_PATH),
            "--param",
            str(MODEL_PATH),
            "--bin",
            str(WEIGHTS_PATH),
            "--image",
            str(IMAGE_PATH),
            "--labels",
            str(LABELS_PATH),
            "--topk",
            "5",
            "--warmup",
            "5",
            "--runs",
            "20",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        print(f"binary exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    stdout = result.stdout
    if "top1_idx=258" not in stdout:
        print(stdout, file=sys.stderr)
        print("expected top1_idx=258", file=sys.stderr)
        return 1
    if "top1_label=Samoyed" not in stdout:
        print(stdout, file=sys.stderr)
        print("expected top1_label=Samoyed", file=sys.stderr)
        return 1

    top5_lines = re.findall(r"^#\d+\s+idx=\d+\s+label=", stdout, flags=re.MULTILINE)
    if len(top5_lines) < 5:
        print(stdout, file=sys.stderr)
        print("expected at least 5 top-k lines", file=sys.stderr)
        return 1

    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
