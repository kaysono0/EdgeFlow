#pragma once

#include "image_loader.h"

#include <memory>
#include <string>
#include <vector>

#include <MNN/Interpreter.hpp>

class MNNClassifier {
public:
    // 加载 MNN 模型并创建 Session。
    //
    // 输入:
    // - model_path: .mnn 模型文件路径。
    // - thread_count: CPU 后端线程数；请求 Metal 时该参数通常不是主要瓶颈，但保留统一接口。
    // - use_metal: true 表示请求 MNN_FORWARD_METAL，false 表示请求 MNN_FORWARD_CPU。
    //
    // 输出:
    // - true: Interpreter 和 Session 创建成功。
    // - false: 模型加载或 Session 创建失败，错误已写入 stderr。
    //
    // 工程目的:
    // - Interpreter 像“模型对象”，Session 像“执行计划”。
    // - 后续做端侧优化时，load 阶段和 classify 阶段要分开看，因为前者包含图优化和内存规划。
    bool load(const std::string& model_path, int thread_count, bool use_metal);

    // 对一张已 resize 的 RGBA 图片执行分类。
    //
    // 输入:
    // - rgba_image: 宽高必须等于模型输入尺寸，像素连续 RGBA 排列。
    //
    // 输出:
    // - softmax 后的概率数组，长度通常是 1000，对应 ImageNet 类别。
    //
    // 工程目的:
    // - 这里串起端侧推理的核心链路：像素 -> 输入 tensor -> runSession -> 输出 tensor -> 后处理。
    // - 分类版本保持简单，后续检测项目会把最后的 softmax 替换为 decode + NMS。
    std::vector<float> classify(const RgbaImage& rgba_image);

    // 从 MNN 输入 tensor 读取模型期望尺寸，避免在代码里写死 224x224。
    int input_width() const;
    int input_height() const;

private:
    std::unique_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
};
