# EdgeFlow

EdgeFlow is a repository for ONNX inference demos and edge deployment experiments.

The current project entry point is:

- `mobilenetv2-demo/`: MobileNetV2 ONNX Runtime demo on macOS, including model analysis, image inference, and CoreML/CPU benchmark scripts

## Quick Start

Enter the demo directory and install dependencies:

```bash
cd mobilenetv2-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the release assets before running the demo:

```bash
python scripts/download_release_assets.py \
  --base-url https://github.com/kaysono0/EdgeFlow/releases/download/v0.1.0
```

Then run:

```bash
python scripts/analyze_model.py
python scripts/mobilenetv2_onnx_demo.py
python scripts/benchmark.py
```

## Release Assets

Large ONNX binaries are distributed through GitHub Releases instead of git history.

Expected assets:

- `mobilenetv2.onnx`
- `mobilenetv2.onnx.data`
- `mobilenetv2_sim.onnx`
- `resnet18.onnx`
- `resnet18.onnx.data`

## Repository Layout

```text
.
├── README.md
├── mobilenetv2-demo
│   ├── README.md
│   ├── models
│   ├── release-assets.json
│   ├── requirements.txt
│   ├── requirements-export.txt
│   └── scripts
└── .gitignore
```

See `mobilenetv2-demo/README.md` for project-specific details.
