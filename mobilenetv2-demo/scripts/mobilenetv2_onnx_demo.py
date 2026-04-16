#!/usr/bin/env python3
"""
Torchvision 分类模型 ONNX 推理示例
=================================
功能:
1. 从 torchvision 导出分类模型为 ONNX 格式
2. 使用 ONNX Runtime 执行推理
3. 解析输出结果并显示 Top-5 分类
4. 使用 Netron 可视化模型结构

使用方法:
    python scripts/mobilenetv2_onnx_demo.py
"""

import inspect
import os
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import urllib

# ======================== 配置 ========================
MODEL_NAME = "mobilenetv2"  # 支持: mobilenetv2, resnet18
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "models"
ONNX_MODEL_PATH = OUTPUT_DIR / f"{MODEL_NAME}.onnx"
ONNX_SIMPLIFIED_PATH = OUTPUT_DIR / f"{MODEL_NAME}_simplified.onnx"
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
TEST_IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
TEST_IMAGE_PATH = OUTPUT_DIR / "test_dog.jpg"

def ensure_onnx_model_exists(onnx_path):
    if os.path.exists(onnx_path):
        return str(onnx_path)

    print(f"⚠️  本地未找到 ONNX 模型: {onnx_path}")
    print("💡 请先运行 `python scripts/download_release_assets.py` 下载 release 资产")
    print("💡 或安装 requirements-export.txt 后重新运行以导出新模型")
    return None


# ======================== 工具函数 ========================
def get_imagenet_labels():
    """下载 ImageNet 分类标签"""
    labels_path = OUTPUT_DIR / "imagenet_classes.txt"
    if not labels_path.exists():
        print("📥 下载 ImageNet 标签...")
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, str(labels_path))
    
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def download_test_image():
    """下载测试图片"""
    if not TEST_IMAGE_PATH.exists():
        print(f"📥 下载测试图片到 {TEST_IMAGE_PATH}...")
        urllib.request.urlretrieve(TEST_IMAGE_URL, str(TEST_IMAGE_PATH))
    return str(TEST_IMAGE_PATH)


def preprocess_image(image_path):
    """预处理图片 (ImageNet 标准预处理)"""
    image = Image.open(image_path).convert('RGB')

    # Match torchvision's Resize(256) + CenterCrop(224) behavior for a PIL image.
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int(round(height * 256 / width))
    else:
        new_height = 256
        new_width = int(round(width * 256 / height))
    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    left = (new_width - 224) // 2
    top = (new_height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    input_array = np.asarray(image, dtype=np.float32) / 255.0
    input_array = np.transpose(input_array, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    input_array = (input_array - mean) / std
    return np.expand_dims(input_array, axis=0)


# ======================== 步骤 1: 导出 ONNX 模型 ========================
def load_torchvision_model(model_name):
    """根据 MODEL_NAME 加载对应的 torchvision 预训练模型"""
    import torchvision.models as models

    model_builders = {
        "mobilenetv2": (
            models.mobilenet_v2,
            models.MobileNet_V2_Weights.IMAGENET1K_V1,
        ),
        "resnet18": (
            models.resnet18,
            models.ResNet18_Weights.IMAGENET1K_V1,
        ),
    }

    if model_name not in model_builders:
        supported_models = ", ".join(model_builders)
        raise ValueError(
            f"不支持的 MODEL_NAME: {model_name}，当前支持: {supported_models}"
        )

    model_builder, weights = model_builders[model_name]
    return model_builder(weights=weights)


def get_onnx_export_compat_kwargs():
    """兼容不同 PyTorch 版本的 ONNX 导出参数名。"""
    import torch

    export_signature = inspect.signature(torch.onnx.export)

    if "dynamo" in export_signature.parameters:
        return {"dynamo": False}
    if "dynamo_export" in export_signature.parameters:
        return {"dynamo_export": False}
    return {}


def export_model_to_onnx():
    """将当前选择的 torchvision 模型导出为 ONNX 格式"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if ONNX_MODEL_PATH.exists():
        print(f"✅ ONNX 模型已存在: {ONNX_MODEL_PATH}")
        return str(ONNX_MODEL_PATH)

    print(f"⚠️  本地未找到 ONNX 模型: {ONNX_MODEL_PATH}")
    try:
        import torch
    except ModuleNotFoundError:
        print("💡 请先运行 `python scripts/download_release_assets.py` 下载 release 资产")
        print("💡 或安装 requirements-export.txt 后重新运行以导出新模型")
        return None
    
    print("\n" + "="*60)
    print(f"步骤 1: 导出 {MODEL_NAME} 为 ONNX 格式")
    print("="*60)
    
    # 加载预训练模型
    print(f"📦 加载 {MODEL_NAME} 预训练模型...")
    model = load_torchvision_model(MODEL_NAME)
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 导出为 ONNX
    print(f"💾 导出 ONNX 模型到 {ONNX_MODEL_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=15,  # 兼容移动端和旧版推理引擎
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        # 关键：禁用新版 Dynamo 导出器，避免自动升级 opset
        **get_onnx_export_compat_kwargs()
    )

    # 验证 ONNX 模型
    onnx_model = onnx.load(str(ONNX_MODEL_PATH))
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX 模型导出成功 (Opset 15): {os.path.getsize(ONNX_MODEL_PATH) / 1024 / 1024:.2f} MB")
    
    return str(ONNX_MODEL_PATH)


# ======================== 步骤 2: ONNX Runtime 推理 ========================
def softmax(x):
    """手动计算 softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def run_onnx_inference(image_path, onnx_path):
    """使用 ONNX Runtime 执行推理"""
    print("\n" + "="*60)
    print("步骤 2: 使用 ONNX Runtime 执行推理")
    print("="*60)
    
    # 加载模型
    print(f"🔧 加载 ONNX 模型: {onnx_path}")
    session = ort.InferenceSession(onnx_path)
    
    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    
    print(f"📐 输入: {input_name}, 形状: {input_shape}")
    print(f"📐 输出: {output_name}, 形状: {output_shape}")
    
    # 预处理图片
    print(f"🖼️  预处理图片: {image_path}")
    input_data = preprocess_image(image_path)
    
    # 执行推理
    print("🚀 执行推理...")
    output = session.run([output_name], {input_name: input_data})[0]
    
    # 解析结果
    probabilities = softmax(output[0])
    top5_indices = np.argsort(probabilities)[::-1][:5]
    
    return top5_indices, probabilities


# ======================== 步骤 3: 显示结果 ========================
def display_results(top5_indices, probabilities, labels):
    """显示 Top-5 分类结果"""
    print("\n" + "="*60)
    print("步骤 3: 推理结果 (Top-5)")
    print("="*60)
    
    for i, idx in enumerate(top5_indices):
        prob = probabilities[idx] * 100
        label = labels[idx]
        bar = "█" * int(prob / 2)
        print(f"{i+1}. {label:30s} {prob:5.2f}% {bar}")


# ======================== 步骤 4: Netron 可视化 (可选) ========================
def launch_netron(onnx_path):
    """启动 Netron 可视化 (需要手动打开浏览器)"""
    import netron
    
    print("\n" + "="*60)
    print("步骤 4: 启动 Netron 可视化")
    print("="*60)
    print(f"🌐 在浏览器中打开: http://localhost:8080")
    print(f"📊 查看模型: {onnx_path}")
    print("⚠️  按 Ctrl+C 停止 Netron 服务")
    
    netron.start(onnx_path, address=8080, browse=True)


# ======================== 步骤 5: 性能基准测试 ========================
def benchmark_inference(onnx_path, num_runs=100):
    """性能基准测试"""
    print("\n" + "="*60)
    print("步骤 5: 性能基准测试")
    print("="*60)
    
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    import time
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
    
    # 正式测试
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # 转换为 ms
    
    times = np.array(times)
    print(f"📊 推理 {num_runs} 次性能统计:")
    print(f"   平均: {times.mean():.2f} ms")
    print(f"   中位数: {np.median(times):.2f} ms")
    print(f"   最小: {times.min():.2f} ms")
    print(f"   最大: {times.max():.2f} ms")
    print(f"   P95: {np.percentile(times, 95):.2f} ms")


# ======================== 主函数 ========================
def main():
    print("\n" + "="*60)
    print(f"🚀 {MODEL_NAME} ONNX 推理示例")
    print("="*60)
    
    # 步骤 1: 导出 ONNX 模型
    onnx_path = export_model_to_onnx()
    if not onnx_path:
        return
    
    # 步骤 2: 准备测试数据
    labels = get_imagenet_labels()
    image_path = download_test_image()
    
    # 步骤 3: 执行推理
    top5_indices, probabilities = run_onnx_inference(image_path, onnx_path)
    
    # 步骤 4: 显示结果
    display_results(top5_indices, probabilities, labels)
    
    # 步骤 5: 性能基准测试
    benchmark_inference(onnx_path, num_runs=50)
    
    print("\n" + "="*60)
    print("✅ 推理完成!")
    print("="*60)
    print("\n💡 提示:")
    print("   1. 运行 `python -c \"import netron; netron.start('models/mobilenetv2.onnx')\"` 查看模型结构")
    print("   2. 继续学习下一课: ONNX 模型简化和量化")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
