import onnxruntime as ort
import numpy as np
import time
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_model_exists(model_path):
    if os.path.exists(model_path):
        return True

    print(f"❌ 找不到模型: {model_path}")
    print("💡 请先运行 `python scripts/download_release_assets.py` 下载 release 资产")
    return False

def benchmark_model(model_path, provider_name, provider_options):
    """
    针对特定后端进行性能测试
    """
    model_name = os.path.basename(model_path)
    print(f"🚀 正在测试: {model_name} | 后端: {provider_name}")
    print("-" * 40)

    # 1. 尝试加载模型
    try:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # 关闭日志
        
        # 初始化 Session 并指定后端
        session = ort.InferenceSession(
            model_path, 
            sess_options, 
            providers=provider_options
        )
        # 获取实际使用的后端 (有时会 Fallback)
        actual_provider = session.get_providers()[0]
        print(f"🔧 实际使用后端: {actual_provider}")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

    input_name = session.get_inputs()[0].name
    input_shape = [1 if isinstance(d, str) else d for d in session.get_inputs()[0].shape]
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # 2. Warmup
    print("🔥 正在预热 (5次)...")
    for _ in range(5):
        session.run(None, {input_name: dummy_input})

    # 3. 正式测试 (100 次)
    runs = 100
    latencies = []
    print(f"⏱️  正在执行 {runs} 次推理...")
    
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    # 4. 统计
    latencies.sort()
    median = latencies[len(latencies) // 2]
    avg = sum(latencies) / len(latencies)
    fps = 1000.0 / median if median > 0 else 0

    return {
        'provider': provider_name,
        'median': median,
        'avg': avg,
        'fps': fps
    }

if __name__ == "__main__":
    # 注意：使用正确的模型名称
    model_path = str(REPO_ROOT / "models" / "mobilenetv2.onnx")
    if not ensure_model_exists(model_path):
        exit()

    print("=" * 60)
    print("🔬 ONNX Runtime Mac 多后端性能对比测试")
    print("=" * 60)
    print()

    # 定义要测试的后端列表 (优先级从高到低)
    # CoreML 会自动利用 GPU/ANE，如果算子不支持会 Fallback 到 CPU
    test_providers = [
        ("CoreML", ['CoreMLExecutionProvider', 'CPUExecutionProvider']),
        ("CPU",    ['CPUExecutionProvider'])
    ]

    results = []

    for name, options in test_providers:
        res = benchmark_model(model_path, name, options)
        if res:
            results.append(res)
        print()

    # 打印对比报告
    if results:
        print("=" * 60)
        print("📊 性能对比报告 (MobileNetV2)")
        print("=" * 60)
        print(f"{'后端 (Provider)':<20} | {'中位数耗时':<12} | {'FPS':<10}")
        print("-" * 60)
        
        cpu_time = 0
        coreml_time = 0
        
        for res in results:
            print(f"{res['provider']:<20} | {res['median']:.2f} ms    | {res['fps']:.1f}")
            if res['provider'] == 'CPU':
                cpu_time = res['median']
            elif res['provider'] == 'CoreML':
                coreml_time = res['median']
                
        print("-" * 60)
        
        # 计算加速比
        if cpu_time > 0 and coreml_time > 0:
            speedup = cpu_time / coreml_time
            print(f"💡 CoreML 相比 CPU 加速: {speedup:.1f}x")
            
            if speedup < 1.0:
                print("⚠️  注意: CoreML 耗时比 CPU 长。")
                print("   原因: MobileNetV2 模型极小，CPU 已能瞬间算完。")
                print("   CoreML 的调度开销 (GPU/ANE 启动延迟) 抵消了计算优势。")
                print("   这在端侧小模型测试中非常正常。")
            else:
                print("✅ 核心结论: CoreML 成功调用 Apple GPU/ANE 加速！")
        print("=" * 60)
    else:
        print("❌ 没有可用的测试结果。")
