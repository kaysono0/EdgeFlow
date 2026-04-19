#!/usr/bin/env python3
"""
MobileNetV2 量化实践脚本
=======================
功能:
1. 复用第一周导出的 models/mobilenetv2.onnx
2. 生成 FP16 和 INT8 动态量化模型
3. 生成 INT8 静态 PTQ 模型并做正式对比
4. 将 INT8 动态量化保留为兼容性验证项
5. 用本地评估清单做真实图片精度对比
6. 输出 Markdown 报告

这个脚本可以把它想成一条小型量化流水线：
- 先拿到一个已经训练好的 FP32 模型
- 再尝试给它换更轻的“包装”：
  - FP16：像把 32 位大箱子换成 16 位小箱子
  - INT8：像把浮点世界里的连续刻度，压到 8 位整数格子里
- 最后比较三件事：
  - 体积有没有变小
  - 推理有没有变快
  - 真实图片上的预测有没有明显变差

使用方法:
    python3 scripts/mobilenetv2_quantization_demo.py
"""

import csv
from pathlib import Path
import os
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from PIL import Image
from tabulate import tabulate


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

ORIGINAL_MODEL = MODELS_DIR / "mobilenetv2.onnx"
FP16_MODEL = MODELS_DIR / "mobilenetv2_fp16.onnx"
INT8_DYNAMIC_MODEL = MODELS_DIR / "mobilenetv2_int8_dynamic.onnx"
INT8_STATIC_MODEL = MODELS_DIR / "mobilenetv2_int8_static.onnx"
REPORT_PATH = REPO_ROOT / "reports" / "quantization_report.md"
TEST_IMAGE = MODELS_DIR / "test_dog.jpg"
LABELS_FILE = MODELS_DIR / "imagenet_classes.txt"
EVAL_MANIFEST = MODELS_DIR / "eval_manifest.csv"


if not hasattr(onnx, "mapping") and hasattr(onnx, "_mapping"):
    # onnxruntime.quantization 仍依赖旧版 onnx.mapping 接口。
    # 可以把这段理解成“给老工具补一层兼容转接头”，
    # 不改模型逻辑，只是让新版 onnx 也能被旧接口识别。
    onnx.mapping = SimpleNamespace(
        TENSOR_TYPE_TO_NP_TYPE={
            key: value.np_dtype for key, value in onnx._mapping.TENSOR_TYPE_MAP.items()
        }
    )


class ImageNetCalibrationDataReader(CalibrationDataReader):
    """使用本地评估清单提供静态 PTQ 校准数据。"""

    def __init__(self, model_path: Path, samples: List[Tuple[Path, int]]):
        session = ort.InferenceSession(str(model_path))
        input_meta = session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_type = input_meta.type
        self.samples = samples
        self.index = 0
        print(
            "  [校准器] 已创建: "
            f"model={model_path.name}, input_name={self.input_name}, "
            f"input_type={self.input_type}, samples={len(samples)}"
        )

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        # 静态 PTQ 需要先“看一眼”一小批真实样本，再决定每层数值范围。
        # 这就像给量化器一把尺子之前，先拿几件真实物品量一下尺寸，
        # 否则尺子定得太宽或太窄，后面量出来都会失真。
        if self.index >= len(self.samples):
            return None

        image_path, _ = self.samples[self.index]
        self.index += 1
        print(
            "  [校准器] 提供样本: "
            f"{self.index}/{len(self.samples)} -> {image_path.name}"
        )
        input_data = preprocess_image(image_path)

        if self.input_type == "tensor(float16)":
            model_input = input_data.astype(np.float16)
        else:
            model_input = input_data.astype(np.float32)

        return {self.input_name: model_input}

    def rewind(self) -> None:
        self.index = 0


def benchmark_model(model_path: Path, input_data: np.ndarray, runs: int = 100) -> dict:
    """加载模型并返回延迟统计结果。"""
    session = ort.InferenceSession(str(model_path))
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    print(
        "  [Benchmark] 加载模型: "
        f"{model_path.name}, input_name={input_name}, "
        f"input_type={input_meta.type}, input_shape={input_data.shape}, runs={runs}"
    )

    if input_meta.type == "tensor(float16)":
        model_input = input_data.astype(np.float16)
    else:
        model_input = input_data.astype(np.float32)
    print(
        "  [Benchmark] 实际送入数据: "
        f"dtype={model_input.dtype}, shape={model_input.shape}"
    )

    # 预热的作用像“先热车”。
    # 第一次推理往往会掺杂图优化、内存分配等一次性开销，
    # 不先预热，第一次耗时会把统计结果带偏。
    for _ in range(10):
        session.run(None, {input_name: model_input})

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: model_input})
        times.append(time.perf_counter() - start)

    return {
        "median": np.median(times) * 1000,
        "p95": np.percentile(times, 95) * 1000,
        "mean": np.mean(times) * 1000,
    }


def load_imagenet_labels() -> List[str]:
    """加载 ImageNet 类别标签。"""
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def preprocess_image(image_path: Path) -> np.ndarray:
    """复用第一周相同的 ImageNet 预处理。"""
    print(f"  [Preprocess] 读取图片: {image_path}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"  [Preprocess] 原始尺寸: {width}x{height}")

    short_side = 256
    if width < height:
        new_width = short_side
        new_height = int(height * short_side / width)
    else:
        new_height = short_side
        new_width = int(width * short_side / height)

    # 先把短边缩放到 256，像是先把不同尺寸的照片统一到一个大致可比较的尺度。
    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    print(f"  [Preprocess] Resize 后尺寸: {new_width}x{new_height}")

    left = (new_width - 224) // 2
    top = (new_height - 224) // 2
    # 再做中心裁剪，得到模型真正吃进去的 224x224 视野。
    # 可以把它理解成“给模型一个固定大小的观察窗口”。
    image = image.crop((left, top, left + 224, top + 224))
    print("  [Preprocess] CenterCrop 后尺寸: 224x224")

    image_array = np.asarray(image).astype(np.float32) / 255.0
    image_array = image_array.transpose(2, 0, 1)

    # 归一化的作用像“统一温度计刻度”。
    # 不同图片原本像素分布不同，减均值、除方差后，
    # 模型看到的输入范围更接近训练时的习惯。
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array - mean) / std

    batched = np.expand_dims(image_array, axis=0)
    print(
        "  [Preprocess] 输出张量: "
        f"shape={batched.shape}, dtype={batched.dtype}"
    )
    return batched


def run_classification(model_path: Path, input_data: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """对单张图片执行分类并返回 Top-5。"""
    session = ort.InferenceSession(str(model_path))
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    print(
        "  [Infer] 加载模型: "
        f"{model_path.name}, input_name={input_meta.name}, "
        f"input_type={input_meta.type}, output_name={output_name}"
    )

    if input_meta.type == "tensor(float16)":
        model_input = input_data.astype(np.float16)
    else:
        model_input = input_data.astype(np.float32)
    print(
        "  [Infer] 实际送入数据: "
        f"dtype={model_input.dtype}, shape={model_input.shape}"
    )

    logits = session.run([output_name], {input_meta.name: model_input})[0][0]
    # 这里手动做 softmax，把模型原始 logits 转成概率。
    # logits 可以理解成每个类别的“打分”，
    # softmax 则把这些打分压成总和为 1 的“投票占比”。
    probabilities = np.exp(logits - np.max(logits))
    probabilities = probabilities / probabilities.sum()
    top5_indices = np.argsort(probabilities)[::-1][:5]
    top5 = [(int(idx), float(probabilities[idx])) for idx in top5_indices]
    print(
        "  [Infer] 输出结果: "
        f"logits_shape={logits.shape}, top1_idx={top5[0][0]}, "
        f"top1_prob={top5[0][1] * 100:.2f}%"
    )
    return logits, top5


def ensure_eval_manifest() -> None:
    """提供一个可直接扩展的本地评估清单。"""
    if EVAL_MANIFEST.exists() or not TEST_IMAGE.exists():
        return

    with open(EVAL_MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        writer.writerow([TEST_IMAGE.name, "258"])


def load_eval_samples() -> Tuple[List[Tuple[Path, int]], List[str]]:
    """加载本地评估清单。"""
    notes: List[str] = []
    print(f"\n📋 读取评估清单: {EVAL_MANIFEST}")

    if not EVAL_MANIFEST.exists():
        return [], [f"缺少 {EVAL_MANIFEST.name}，跳过批量精度对比。"]

    samples: List[Tuple[Path, int]] = []
    with open(EVAL_MANIFEST, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            image_value = (row.get("image") or "").strip()
            label_value = (row.get("label") or "").strip()

            if not image_value or not label_value:
                notes.append(f"{EVAL_MANIFEST.name}:{row_num} 缺少 image 或 label，已跳过。")
                continue

            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = MODELS_DIR / image_path

            if not image_path.exists():
                notes.append(f"{EVAL_MANIFEST.name}:{row_num} 图片不存在: {image_path}")
                continue

            try:
                label_idx = int(label_value)
            except ValueError:
                notes.append(f"{EVAL_MANIFEST.name}:{row_num} label 不是整数: {label_value}")
                continue

            samples.append((image_path, label_idx))

    print(f"  [Manifest] 有效样本数: {len(samples)}")
    for image_path, label_idx in samples:
        print(f"  [Manifest] 样本: image={image_path.name}, label={label_idx}")

    return samples, notes


def compare_real_image_predictions() -> Tuple[List[List[str]], List[List[str]], List[str]]:
    """对本地评估清单做 FP32 / FP16 精度对比。"""
    if not LABELS_FILE.exists():
        return [], [], ["缺少 imagenet_classes.txt，跳过真实图片精度对比。"]

    labels = load_imagenet_labels()
    samples, notes = load_eval_samples()
    if not samples:
        if not notes:
            notes.append("评估清单为空，跳过真实图片精度对比。")
        return [], [], notes

    detail_rows: List[List[str]] = []
    summary_rows: List[List[str]] = []
    # 这里把 FP32 当成“老师版本”，其他量化模型都拿它来对照。
    # 因为我们的目标不是证明量化模型绝对正确，
    # 而是看它有没有明显偏离原始 FP32 基线。
    model_variants = [("FP32", ORIGINAL_MODEL), ("FP16", FP16_MODEL)]
    if INT8_STATIC_MODEL.exists():
        model_variants.append(("INT8 (静态 PTQ)", INT8_STATIC_MODEL))

    summary_stats = {
        label: {"correct": 0, "total": 0, "fp32_agree": 0, "cosines": []}
        for label, _ in model_variants
    }

    for image_path, label_idx in samples:
        print(
            "\n  [Eval] 开始评估样本: "
            f"image={image_path.name}, label_idx={label_idx}, label={labels[label_idx]}"
        )
        input_data = preprocess_image(image_path)
        fp32_logits, fp32_top5 = run_classification(ORIGINAL_MODEL, input_data)
        fp32_top1 = fp32_top5[0][0]
        print(
            "  [Eval] FP32 基线 Top-1: "
            f"{labels[fp32_top1]} ({fp32_top5[0][1] * 100:.2f}%)"
        )

        for label, model_path in model_variants:
            try:
                logits, top5 = run_classification(model_path, input_data)
                top1_idx = top5[0][0]
                # cosine similarity 像是在比较两张箭头图的朝向是否一致。
                # 越接近 1，说明量化后的输出方向越像 FP32，
                # 即使每个数不完全一样，整体判断倾向也仍然接近。
                cosine = np.dot(fp32_logits, logits) / (
                    np.linalg.norm(fp32_logits) * np.linalg.norm(logits)
                )
                summary_stats[label]["total"] += 1
                summary_stats[label]["correct"] += int(top1_idx == label_idx)
                summary_stats[label]["fp32_agree"] += int(top1_idx == fp32_top1)
                summary_stats[label]["cosines"].append(float(cosine))
                detail_rows.append(
                    [
                        image_path.name,
                        label,
                        labels[label_idx],
                        labels[top1_idx],
                        f"{top5[0][1] * 100:.2f}%",
                        "是" if top1_idx == label_idx else "否",
                        "是" if top1_idx == fp32_top1 else "否",
                        f"{cosine:.6f}",
                    ]
                )
                print(
                    "  [Eval] 模型结果: "
                    f"model={label}, top1={labels[top1_idx]}, "
                    f"prob={top5[0][1] * 100:.2f}%, "
                    f"correct={'是' if top1_idx == label_idx else '否'}, "
                    f"agree_fp32={'是' if top1_idx == fp32_top1 else '否'}, "
                    f"cosine={cosine:.6f}"
                )
            except Exception as exc:
                notes.append(f"{label} 在 {image_path.name} 上推理失败: {exc}")

        try:
            _, int8_top5 = run_classification(INT8_DYNAMIC_MODEL, input_data)
            notes.append(
                f"INT8 (动态) 在 {image_path.name} 上 Top-1: "
                f"{labels[int8_top5[0][0]]} ({int8_top5[0][1] * 100:.2f}%)"
            )
        except Exception as exc:
            notes.append(f"INT8 (动态) 在 {image_path.name} 上推理失败: {exc}")

    for model_name, stats in summary_stats.items():
        if stats["total"] == 0:
            continue
        summary_rows.append(
            [
                model_name,
                str(stats["total"]),
                f"{stats['correct'] / stats['total'] * 100:.2f}%",
                f"{stats['fp32_agree'] / stats['total'] * 100:.2f}%",
                f"{np.mean(stats['cosines']):.6f}",
            ]
        )

    return summary_rows, detail_rows, notes


def ensure_original_model() -> None:
    """确认第一周导出的 ONNX 模型存在。"""
    if ORIGINAL_MODEL.exists():
        print(f"✅ 原始模型存在: {ORIGINAL_MODEL}")
        return

    print(f"❌ 未找到模型: {ORIGINAL_MODEL}")
    print("请先运行 `python3 scripts/mobilenetv2_onnx_demo.py` 导出 ONNX 模型。")
    raise SystemExit(1)


def export_fp16_model() -> int:
    """将 FP32 模型转换为 FP16。"""
    try:
        from onnxconverter_common import float16
    except ModuleNotFoundError:
        print("❌ 缺少依赖: onnxconverter_common")
        print(
            "请先安装: python3 -m pip install onnxconverter-common"
        )
        print(
            "如果你使用 ai-deploy 环境，请改用: conda run -n ai-deploy python -m pip install onnxconverter-common"
        )
        raise SystemExit(1)

    print("\n📊 步骤 1: 生成 FP16 模型...")
    print(f"  输入模型: {ORIGINAL_MODEL}")
    print(f"  输出模型: {FP16_MODEL}")
    model = onnx.load(str(ORIGINAL_MODEL))
    # FP16 像是“轻量化压箱”。
    # 箱子更小了，但里面装的仍然是浮点数，不像 INT8 那样进入整数世界。
    # 所以它通常比 INT8 更稳，但收益也未必最大。
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, str(FP16_MODEL))
    fp16_size = os.path.getsize(FP16_MODEL)
    print(f"  已保存: {FP16_MODEL}")
    print(f"  大小: {fp16_size / 1e6:.2f} MB")
    return fp16_size


def export_int8_dynamic_model() -> int:
    """对模型执行 INT8 动态量化。"""
    print("\n📊 步骤 2: 生成 INT8 动态量化模型...")
    print(f"  输入模型: {ORIGINAL_MODEL}")
    print(f"  输出模型: {INT8_DYNAMIC_MODEL}")
    # 动态量化更像“跑的时候再临时决定激活值怎么压缩”。
    # 它上手简单，不需要提前准备校准集，
    # 但对卷积网络来说，运行时兼容性不一定理想。
    #
    # 下面这个 quantize_dynamic(...) 调用，可以把它理解成：
    # “先把模型里的固定货物（主要是权重）压缩打包，
    #  真正推理时再根据现场情况决定活动部件（激活值）怎么量尺寸。”
    #
    # 参数逐个解释：
    # - model_input:
    #   原始 FP32 ONNX 模型路径。它是“待压缩的原箱子”。
    # - model_output:
    #   输出的动态量化模型路径。它是“压缩后的新箱子”。
    # - weight_type=QuantType.QInt8:
    #   把权重量化成有符号 8 位整数。权重里既有正数也有负数，
    #   所以用 QInt8 比较自然。
    # - per_channel=True:
    #   每个输出通道单独使用一把尺子（一个 scale），
    #   而不是整块权重共用一把尺子。好比给不同尺寸的衣服分别量身，
    #   通常比“所有人穿同一件均码”更稳，精度更容易保住。
    # - reduce_range=False:
    #   不主动缩小整数范围，尽量用满 INT8 的表示空间。
    #   如果设成 True，等于主动少用一部分格子，兼容性有时更稳，
    #   但可表示的细节会再少一点。
    #
    # 这里没有显式传 activation_type。
    # 原因是动态量化的重点通常是“先压缩权重”，
    # 激活值的量化参数更多是在运行时临时决定。
    # 对 MobileNetV2 这种卷积网络，动态量化更适合作为“兼容性验证项”，
    # 而不是当前这份实验里的正式主结论。
    quantize_dynamic(
        model_input=str(ORIGINAL_MODEL),
        model_output=str(INT8_DYNAMIC_MODEL),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    int8_size = os.path.getsize(INT8_DYNAMIC_MODEL)
    print(f"  已保存: {INT8_DYNAMIC_MODEL}")
    print(f"  大小: {int8_size / 1e6:.2f} MB")
    return int8_size


def export_int8_static_model(samples: List[Tuple[Path, int]]) -> Optional[int]:
    """用本地样本执行静态 PTQ。"""
    if not samples:
        print("\n📊 步骤 3: 跳过 INT8 静态 PTQ（没有可用校准样本）...")
        return None

    print("\n📊 步骤 3: 生成 INT8 静态 PTQ 模型...")
    print(f"  输入模型: {ORIGINAL_MODEL}")
    print(f"  输出模型: {INT8_STATIC_MODEL}")
    print(f"  校准样本来源: {EVAL_MANIFEST}")
    calibration_reader = ImageNetCalibrationDataReader(ORIGINAL_MODEL, samples)
    # 静态 PTQ 的核心是“先校准，再压缩”。
    # 可以把它想成先拿一批真实货物测量尺寸，
    # 再统一设计更合适的收纳盒。
    #
    # 这里使用 QDQ（QuantizeLinear + DequantizeLinear）格式。
    # 它像是在原图里插入“压缩”和“还原”的标记点，
    # 告诉后端哪些位置应该按量化方式处理。
    #
    # 参数逐个解释：
    # - model_input / model_output:
    #   分别是原始模型和输出模型路径，含义和动态量化一致。
    # - calibration_data_reader:
    #   校准数据读取器。它像“拿着样品进仓库量尺寸的人”，
    #   会把真实图片送进模型，统计各层激活值范围，
    #   让量化器知道每层应该用多大的 scale / zero_point。
    # - quant_format=QuantFormat.QDQ:
    #   使用 QDQ 图格式。它不是直接把所有算子都改名，
    #   而是在图里显式插入 QuantizeLinear / DequantizeLinear 节点。
    #   这种写法更像在流程图上贴“这里压缩、这里还原”的标签，
    #   对很多后端更友好，也更容易调试。
    # - activation_type=QuantType.QUInt8:
    #   激活值使用无符号 8 位整数。很多激活经过 ReLU/ReLU6 后以非负为主，
    #   用 QUInt8 很常见。
    # - weight_type=QuantType.QInt8:
    #   权重使用有符号 8 位整数，因为卷积核权重通常正负都有。
    # - per_channel=True:
    #   权重按通道量化。对于卷积网络，这是很常见也很重要的配置，
    #   因为不同输出通道的数值范围差异往往不小。
    # - calibrate_method=CalibrationMethod.MinMax:
    #   校准时使用最小值/最大值来决定量化范围。
    #   你可以把它理解成“先看样品里最大和最小的箱子各有多大，
    #   再决定统一货架的上下边界”。
    #   它简单直观，适合入门演示。
    # - reduce_range=False:
    #   和动态量化一样，这里不主动缩小整数表示范围。
    # - extra_options={"ActivationSymmetric": False}:
    #   激活值使用非对称量化。原因是激活分布常常不以 0 为中心，
    #   非对称量化更像“把尺子的零点挪到更合适的位置”，
    #   能更充分利用 8 位整数的格子。
    quantize_static(
        model_input=str(ORIGINAL_MODEL),
        model_output=str(INT8_STATIC_MODEL),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        reduce_range=False,
        extra_options={"ActivationSymmetric": False},
    )
    int8_size = os.path.getsize(INT8_STATIC_MODEL)
    print(f"  已保存: {INT8_STATIC_MODEL}")
    print(f"  大小: {int8_size / 1e6:.2f} MB")
    print(f"  校准样本数: {len(samples)}")
    return int8_size


def save_report(
    results: List[List[str]],
    image_summary: List[List[str]],
    image_details: List[List[str]],
    image_notes: List[str],
    static_ptq_notes: List[str],
    compatibility_notes: List[str],
) -> None:
    """将量化结果写入 Markdown 报告。"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n📝 正在写报告: {REPORT_PATH}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# MobileNetV2 量化对比报告\n\n")
        f.write("## 测试环境\n")
        f.write("- 设备: Mac M1 Pro\n")
        f.write("- 模型: MobileNetV2\n")
        f.write("- 运行时: ONNX Runtime\n")
        f.write("- 测试轮数: 100\n\n")
        f.write("## 正式对比结果\n\n")
        f.write(
            tabulate(
                results,
                headers=["模型", "中位数 (ms)", "P95 (ms)", "均值 (ms)", "模型大小 (MB)"],
                tablefmt="pipe",
            )
        )
        f.write("\n")
        if image_summary:
            f.write("\n## 本地图片集精度汇总\n\n")
            f.write(f"评估清单: `{EVAL_MANIFEST.name}`\n\n")
            f.write(
                tabulate(
                    image_summary,
                    headers=["模型", "样本数", "Top-1 准确率", "与 FP32 Top-1 一致率", "平均 logits cosine"],
                    tablefmt="pipe",
                )
            )
            f.write("\n")

        if image_details:
            f.write("\n## 样本级结果\n\n")
            f.write(
                tabulate(
                    image_details,
                    headers=["图片", "模型", "标签", "Top-1 预测", "Top-1 概率", "预测正确", "与 FP32 Top-1 一致", "logits cosine"],
                    tablefmt="pipe",
                )
            )
            f.write("\n")

        if image_notes:
            f.write("\n## 备注\n\n")
            for note in image_notes:
                f.write(f"- {note}\n")
        if static_ptq_notes:
            f.write("\n## 静态 PTQ 备注\n\n")
            for note in static_ptq_notes:
                f.write(f"- {note}\n")
        if compatibility_notes:
            f.write("\n## 兼容性验证项\n\n")
            for note in compatibility_notes:
                f.write(f"- {note}\n")


def main() -> None:
    print("=" * 60)
    print("🚀 MobileNetV2 量化实践")
    print("=" * 60)

    ensure_eval_manifest()
    ensure_original_model()
    # eval_samples 既承担“精度评估样本”，也承担“静态 PTQ 校准样本”。
    # 这不是最严格的研究做法，但对当前教学实验足够直接：
    # 用同一批真实图片同时完成校准和观察结果。
    eval_samples, eval_notes = load_eval_samples()
    print(f"📦 模型目录: {MODELS_DIR}")
    print(f"🖼️  测试图片默认入口: {TEST_IMAGE}")
    print(f"🏷️  标签文件: {LABELS_FILE}")

    original_size = os.path.getsize(ORIGINAL_MODEL)
    print(f"   大小: {original_size / 1e6:.2f} MB")

    fp16_size = export_fp16_model()
    static_ptq_notes = list(eval_notes)
    compatibility_notes: List[str] = []
    int8_dynamic_size = export_int8_dynamic_model()
    compatibility_notes.append(
        f"INT8 (动态) 模型已生成: {INT8_DYNAMIC_MODEL.name} ({int8_dynamic_size / 1e6:.2f} MB)"
    )
    int8_static_size = None
    try:
        int8_static_size = export_int8_static_model(eval_samples)
    except Exception as exc:
        static_ptq_notes.append(f"INT8 (静态 PTQ) 模型生成失败: {exc}")

    print("\n📊 步骤 4: 性能对比测试...")
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    print(
        "  [Benchmark] 构造随机输入: "
        f"shape={input_data.shape}, dtype={input_data.dtype}"
    )
    # 这里用随机输入做 benchmark，只为了看“模型跑起来有多快”。
    # 它像测发动机空转性能，不代表真实上路表现。
    # 真正的分类质量要看后面的真实图片评估。

    results: List[List[str]] = []
    model_variants = [
        ("FP32", ORIGINAL_MODEL, original_size),
        ("FP16", FP16_MODEL, fp16_size),
    ]
    if int8_static_size is not None:
        model_variants.append(("INT8 (静态 PTQ)", INT8_STATIC_MODEL, int8_static_size))

    for label, model_path, model_size in model_variants:
        try:
            print(
                f"\n  [Benchmark] 开始测试 {label}: "
                f"model={model_path.name}, size={model_size / 1e6:.2f} MB"
            )
            stats = benchmark_model(model_path, input_data)
            results.append(
                [
                    label,
                    f"{stats['median']:.2f}",
                    f"{stats['p95']:.2f}",
                    f"{stats['mean']:.2f}",
                    f"{model_size / 1e6:.2f}",
                ]
            )
            print(
                "  [Benchmark] 完成: "
                f"median={stats['median']:.2f} ms, "
                f"p95={stats['p95']:.2f} ms, mean={stats['mean']:.2f} ms"
            )
        except Exception as exc:
            print(f"  ⚠️  {label} 推理失败: {exc}")

    try:
        benchmark_model(INT8_DYNAMIC_MODEL, input_data)
        compatibility_notes.append("INT8 (动态) 在当前后端可执行。")
    except Exception as exc:
        compatibility_notes.append(f"INT8 (动态) 在当前后端不可执行: {exc}")

    print("\n" + "=" * 60)
    print("📈 正式量化对比结果")
    print("=" * 60)
    print(
        tabulate(
            results,
            headers=["模型", "中位数 (ms)", "P95 (ms)", "均值 (ms)", "模型大小 (MB)"],
            tablefmt="grid",
        )
    )
    print("=" * 60)

    print("\n📊 步骤 5: 本地图片集精度对比...")
    # 到这一步才是在看“这辆车上路是否还稳”。
    # 前面的体积和延迟是工程收益，
    # 这里的 Top-1 和 cosine 则是在看量化有没有把模型判断带偏。
    image_summary, image_details, image_notes = compare_real_image_predictions()
    if image_summary:
        print(
            tabulate(
                image_summary,
                headers=["模型", "样本数", "Top-1 准确率", "与 FP32 Top-1 一致率", "平均 logits cosine"],
                tablefmt="grid",
            )
        )
    if image_details:
        print(
            tabulate(
                image_details,
                headers=["图片", "模型", "标签", "Top-1 预测", "Top-1 概率", "预测正确", "与 FP32 Top-1 一致", "logits cosine"],
                tablefmt="grid",
            )
        )
    for note in image_notes:
        print(f"  ℹ️  {note}")
    for note in static_ptq_notes:
        print(f"  ℹ️  {note}")
    for note in compatibility_notes:
        print(f"  ℹ️  {note}")

    save_report(
        results,
        image_summary,
        image_details,
        image_notes,
        static_ptq_notes,
        compatibility_notes,
    )
    print(f"\n✅ 报告已保存到: {REPORT_PATH}")
    print("\n下一步建议:")
    print("1. 往 eval_manifest.csv 继续加真实图片，扩大静态 PTQ 校准集")
    print("2. 对比静态 PTQ 和动态量化在目标后端上的真实支持情况")


if __name__ == "__main__":
    main()
