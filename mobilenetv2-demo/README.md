# MobileNetV2 ONNX Demo

This project demonstrates how to inspect and run a MobileNetV2 ONNX model with ONNX Runtime on macOS. It includes:

- ONNX model structure analysis
- ONNX Runtime inference on a local test image
- CoreML and CPU execution provider benchmark comparison

## Project Layout

```text
models/
  test_dog.jpg
  imagenet_classes.txt
scripts/
  analyze_model.py
  mobilenetv2_onnx_demo.py
  benchmark.py
  download_release_assets.py
release-assets.json
```

## Setup

Create a virtual environment and install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The included demo can run inference with the checked-in ONNX model without PyTorch. Install `requirements-export.txt` only if you want to export a fresh model from torchvision.

## Download Release Assets

Large ONNX model files are not tracked in git. Download them from your release assets before running the demo:

```bash
python scripts/download_release_assets.py \
  --base-url https://github.com/kaysono0/EdgeFlow/releases/download/v0.1.0
```

You can also set the base URL once:

```bash
export EDGEFLOW_RELEASE_BASE_URL=https://github.com/kaysono0/EdgeFlow/releases/download/v0.1.0
python scripts/download_release_assets.py
```

Upload these files to the matching release tag:

- `mobilenetv2.onnx`
- `mobilenetv2.onnx.data`
- `mobilenetv2_sim.onnx`
- `resnet18.onnx`
- `resnet18.onnx.data`

## Run

Analyze the simplified ONNX model:

```bash
python scripts/analyze_model.py
```

Run image classification inference and a short benchmark:

```bash
python scripts/mobilenetv2_onnx_demo.py
```

Compare CoreML and CPU execution providers:

```bash
python scripts/benchmark.py
```

## Notes

- The benchmark numbers depend on hardware, thermal state, and ONNX Runtime version.
- `CoreMLExecutionProvider` is available on macOS builds of ONNX Runtime.
- Install `requirements-export.txt` only when you want to export new ONNX files from torchvision locally.
