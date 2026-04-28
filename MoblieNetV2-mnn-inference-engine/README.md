# Week 03 MNN MobileNetV2 Inference

本目录保存一套可运行的 `MobileNetV2 + MNN` 最小闭环。

它的目标不是做一个复杂 demo，而是把这条链路跑成可回归的证据：

- 加载 `assets/models/mobilenetv2.mnn`
- 使用 macOS ImageIO 解码 JPEG，并缩放到模型输入尺寸
- 使用 MNN `ImageProcess` 完成 RGBA -> RGB 和 ImageNet normalize
- 创建 MNN `Session`，支持请求 `cpu` 或 `metal`
- 执行 warmup 后的多轮推理计时
- 输出 softmax top-5、mean、median、p95

## 目录结构

```text
experiments/week03-mnn-inference-engine/
├── CMakeLists.txt
├── README.md
├── include/
├── src/
└── tests/
```

## 构建

```bash
cmake -S experiments/week03-mnn-inference-engine \
      -B experiments/week03-mnn-inference-engine/build \
      -DCMAKE_BUILD_TYPE=Release

cmake --build experiments/week03-mnn-inference-engine/build --config Release
```

如果 MNN 不在默认位置：

```bash
cmake -S experiments/week03-mnn-inference-engine \
      -B experiments/week03-mnn-inference-engine/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DMNN_ROOT=/path/to/MNN \
      -DMNN_BUILD_DIR=/path/to/MNN/build
```

## 运行

### CPU

```bash
experiments/week03-mnn-inference-engine/build/mnn_mobilenetv2 \
    assets/models/mobilenetv2.mnn \
    assets/models/test_dog.jpg \
    cpu \
    assets/models/imagenet_classes.txt
```

### Metal

```bash
experiments/week03-mnn-inference-engine/build/mnn_mobilenetv2 \
    assets/models/mobilenetv2.mnn \
    assets/models/test_dog.jpg \
    metal \
    assets/models/imagenet_classes.txt
```

程序输出 `Requested backend`。如果 MNN 运行时打印 `Can't Find type=1 backend, use 0 instead`，说明当前 MNN 构建或运行环境没有成功启用 Metal，实际回退到了 CPU。

### 带 benchmark 参数

最后两个可选参数是 warmup 次数和测试次数：

```bash
experiments/week03-mnn-inference-engine/build/mnn_mobilenetv2 \
    assets/models/mobilenetv2.mnn \
    assets/models/test_dog.jpg \
    cpu \
    assets/models/imagenet_classes.txt \
    3 \
    10
```

## 验证

### Smoke test

```bash
ctest --test-dir experiments/week03-mnn-inference-engine/build --output-on-failure
```

### 批量性能测试

```bash
python3 experiments/week03-mnn-inference-engine/tests/benchmark_mobilenetv2_mnn.py --backend cpu --warmup 10 --runs 100
```

如果你想同时对比 CPU 和 Metal 请求：

```bash
python3 experiments/week03-mnn-inference-engine/tests/benchmark_mobilenetv2_mnn.py --backend both --warmup 10 --runs 100
```

脚本会读取 `assets/models/eval_manifest.csv`，对仓库里存在的图片逐个运行 `mnn_mobilenetv2`，并输出一张 Markdown 表，包含：

- `top1 idx / label / prob`
- `mean / median / p95`
- `backend` 请求结果

这份脚本的作用是把 MNN baseline 的性能数据固定下来，方便和 NCNN baseline 做横向比较。

脚本位置：

- [`tests/benchmark_mobilenetv2_mnn.py`](/Users/junsen/workspace/LearnAI/experiments/week03-mnn-inference-engine/tests/benchmark_mobilenetv2_mnn.py#L1)

## 说明

- MNN 在 Apple 平台有原生 `Metal` backend
- `--backend both` 会同时测 `CPU` 和 `Metal`
- 如果 `metal` 请求在当前构建或 runtime 中不可用，脚本会保留请求结果并在报告里写明回退情况
- 这份 baseline 主要用于 macOS 开发验证和周报数据固定，不是独立的精度评测
