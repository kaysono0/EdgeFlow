# Week 03 NCNN Conversion

本目录现在保存一套可运行的 `MobileNetV2 + NCNN` 最小闭环：

- 复用 `assets/models/mobilenetv2.onnx` 的转换产物
- 用 C++ 跑 `test_dog.jpg` 的分类推理
- 输出 top-5 和 benchmark
- 用 smoke test 验证 top-1 是否稳定落在 `Samoyed`

## 目录结构

```text
experiments/week03-ncnn-conversion/
├── CMakeLists.txt
├── README.md
├── outputs/
│   ├── mobilenetv2.param
│   └── mobilenetv2.bin
├── src/
│   └── mobilenetv2_ncnn.cpp
└── tests/
    └── verify_mobilenetv2_ncnn.py
```

## 构建

依赖：

- `/Users/junsen/workspace/ncnn/build/install/lib/cmake/ncnn`
- `/Users/junsen/workspace/ncnn/src`

构建命令：

```bash
cd experiments/week03-ncnn-conversion
cmake -S . -B build -DNCNN_DIR=/Users/junsen/workspace/ncnn/build/install/lib/cmake/ncnn
cmake --build build -j4
```

## 运行

```bash
./build/mobilenetv2_ncnn \
  --param outputs/mobilenetv2.param \
  --bin outputs/mobilenetv2.bin \
  --image ../../assets/models/test_dog.jpg \
  --labels ../../assets/models/imagenet_classes.txt \
  --backend cpu \
  --topk 5 \
  --warmup 5 \
  --runs 20
```

如果你的 NCNN 构建启用了 Vulkan，并且当前机器 / 真机运行时可用，也可以显式请求：

```bash
./build/mobilenetv2_ncnn \
  --param outputs/mobilenetv2.param \
  --bin outputs/mobilenetv2.bin \
  --image ../../assets/models/test_dog.jpg \
  --labels ../../assets/models/imagenet_classes.txt \
  --backend vulkan \
  --topk 5 \
  --warmup 5 \
  --runs 20
```

## 验证

运行 smoke test：

```bash
python3 tests/verify_mobilenetv2_ncnn.py
```

验证标准：

- 程序能成功运行
- `top1_idx=258`
- `top1_label=Samoyed`
- 至少打印 5 条 top-k 结果

批量 regression 验证：

```bash
python3 tests/verify_mobilenetv2_ncnn_batch.py --backend cpu --warmup 5 --runs 20
```

它会用 `assets/models/eval_manifest.csv` 里已有的本地图片，和已记录的 baseline 预测做对照。
这不是 ground-truth accuracy 评估，而是用来检查 NCNN 是否相对已有 baseline 发生漂移。
脚本会同时导出报告到 [`reports/week3_ncnn_inference_baseline.md`](/Users/junsen/workspace/LearnAI/reports/week3_ncnn_inference_baseline.md#L1)。

如果你的 NCNN 构建支持 Vulkan，也可以测双后端：

```bash
python3 tests/verify_mobilenetv2_ncnn_batch.py --backend both --warmup 5 --runs 20
```

## 说明

- 当前转换后的输入 blob 是 `in0`
- 输出 blob 是 `out0`
- 预处理按 ImageNet 习惯做了 `resize short side to 256 + center crop 224 + normalize`
- NCNN 没有原生 `Metal backend`；Apple 平台上的 GPU 路线是 `Vulkan`，通常通过 `MoltenVK` 运行
- 当前仓库里的默认 baseline 仍以 CPU 为主，Vulkan 需要 NCNN 的 Vulkan-enabled build 和可用 runtime
