import onnx
import numpy as np
from collections import Counter
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_file_exists(model_path):
    if os.path.exists(model_path):
        return True

    print(f"❌ 错误: 找不到模型文件 {model_path}")
    print("💡 请先运行 `python scripts/download_release_assets.py` 下载 release 资产")
    return False

def analyze_model(model_path):
    """
    分析 ONNX 模型结构、算子统计和参数量
    """
    if not ensure_file_exists(model_path):
        return

    print(f"📂 正在分析: {model_path}")
    print("-" * 40)

    # 1. 加载模型
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 2. 检查模型合法性
    try:
        onnx.checker.check_model(model)
        print("✅ 模型格式验证通过 (Checker OK)")
    except Exception as e:
        print(f"⚠️  模型格式警告: {e}")

    # 3. 分析输入输出
    print("\n📥 输入信息:")
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value != 0:
                shape.append(dim.dim_value)
            else:
                # 动态维度 (如 batch_size)
                shape.append(dim.dim_param if dim.dim_param else "?")
        print(f"  名称: {inp.name:<15} | 形状: {shape} | 类型: {inp.type.tensor_type.elem_type}")

    print("\n📤 输出信息:")
    for out in model.graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            if dim.dim_value != 0:
                shape.append(dim.dim_value)
            else:
                shape.append(dim.dim_param if dim.dim_param else "?")
        print(f"  名称: {out.name:<15} | 形状: {shape} | 类型: {out.type.tensor_type.elem_type}")

    # 4. 统计节点 (算子) - 影响推理调度开销
    print("\n⚙️  算子统计 (Nodes):")
    op_counts = Counter()
    for node in model.graph.node:
        op_counts[node.op_type] += 1
    
    # 按数量倒序排列
    total_nodes = len(model.graph.node)
    print(f"  总节点数: {total_nodes}")
    print("  Top 5 算子:")
    for op, count in op_counts.most_common(5):
        percentage = (count / total_nodes) * 100
        bar = "█" * int(percentage / 2)
        print(f"    {op:<20} {count:>4} ({percentage:.1f}%) {bar}")

    # 5. 统计参数量 (Initializers/Weights) - 影响模型体积和内存
    print("\n💾 参数量统计 (Weights):")
    total_params = 0
    param_types = Counter()
    
    for init in model.graph.initializer:
        # 计算参数量 = 所有维度乘积
        dims = [d for d in init.dims]
        count = int(np.prod(dims)) if dims else 0
        total_params += count
        param_types[init.data_type] += 1

    print(f"  总参数量: {total_params:,}")
    print(f"  模型体积预估: {total_params * 4 / 1024 / 1024:.2f} MB (假设 Float32)")
    
    # 常见数据类型映射
    type_map = {1: "Float32", 7: "Int64", 6: "Int32", 10: "Float16"}
    print("  权重类型分布:")
    for dtype, count in param_types.most_common():
        type_name = type_map.get(dtype, f"Type_{dtype}")
        print(f"    {type_name}: {count} 个张量")

if __name__ == "__main__":
    # 分析 MobileNetV2
    analyze_model(str(REPO_ROOT / "models" / "mobilenetv2_sim.onnx"))
    
    # 如果你有 ResNet18 也可以解开注释对比看看
    # analyze_model(str(REPO_ROOT / "models" / "resnet18.onnx"))
