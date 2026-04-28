#include "mnn_classifier.h"

#include "infer_utils.h"

#include <MNN/ImageProcess.hpp>
#include <MNN/Tensor.hpp>

#include <cstring>
#include <iostream>
#include <stdexcept>

bool MNNClassifier::load(const std::string& model_path, int thread_count, bool use_metal) {
    // Interpreter 持有模型结构和权重，是 MNN 推理链路的入口。
    interpreter_.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!interpreter_) {
        std::cerr << "Failed to load MNN model: " << model_path << "\n";
        return false;
    }

    MNN::ScheduleConfig config;
    config.numThread = thread_count;
    config.type = use_metal ? MNN_FORWARD_METAL : MNN_FORWARD_CPU;

    // Session 是推理引擎生成的执行计划。
    // 创建 Session 时，MNN 会做后端选择、内存规划和部分图优化；
    // 因此它属于初始化成本，不应该混进单帧推理耗时。
    session_ = interpreter_->createSession(config);
    if (!session_) {
        std::cerr << "Failed to create MNN session\n";
        return false;
    }

    return true;
}

std::vector<float> MNNClassifier::classify(const RgbaImage& rgba_image) {
    if (!interpreter_ || !session_) {
        throw std::runtime_error("classifier is not loaded");
    }
    if (rgba_image.width != input_width() || rgba_image.height != input_height()) {
        throw std::runtime_error("image size does not match model input size");
    }

    // 从 Session 获取输入 tensor，准备把图片数据写入这个 tensor。
    // 这里不直接在 load 阶段获取输入 tensor，因为有些模型可能在 Session 创建时才确定输入输出 tensor 的形状和类型。
    MNN::Tensor* input = interpreter_->getSessionInput(session_, nullptr);
    if (!input) {
        throw std::runtime_error("failed to get MNN input tensor");
    }

    // MNN ImageProcess 负责把 RGBA byte buffer 转换成模型输入 tensor。
    // config 里指定了颜色通道转换和 mean/std 归一化，因此业务侧不需要关心这些细节。
    MNN::CV::ImageProcess::Config image_config;

    // image_loader 输出 RGBA，而 MobileNetV2 训练预处理通常基于 RGB。
    // ImageProcess 在写入 tensor 时完成 RGBA -> RGB 和 uint8 -> float 的转换。
    image_config.sourceFormat = MNN::CV::RGBA;
    image_config.destFormat = MNN::CV::RGB;

    // 这组 mean/normal 对应 PyTorch ImageNet 常用预处理:
    // normalized = (pixel - mean) / std。
    // 这里 std 已经换算成 1/std，避免每个像素做除法。
    const float mean[3] = {123.675f, 116.28f, 103.53f};
    const float normal[3] = {1.0f / 58.395f, 1.0f / 57.12f, 1.0f / 57.375f};
    std::memcpy(image_config.mean, mean, sizeof(mean));
    std::memcpy(image_config.normal, normal, sizeof(normal));

    std::unique_ptr<MNN::CV::ImageProcess> process(
        MNN::CV::ImageProcess::create(image_config));
    if (!process) {
        throw std::runtime_error("failed to create MNN image processor");
    }

    // stride 使用 width * 4，因为 RGBA 每个像素占 4 字节且按行连续存放。
    // convert 会把结果直接写入 MNN 输入 tensor，避免业务侧手动处理 NCHW/NHWC 细节。
    process->convert(
        rgba_image.pixels.data(),
        rgba_image.width,
        rgba_image.height,
        rgba_image.width * 4,
        input);

    // 真正执行计算图。CPU/Metal 等后端选择已经体现在 session 的执行计划里。
    interpreter_->runSession(session_);

    MNN::Tensor* output = interpreter_->getSessionOutput(session_, nullptr);
    if (!output) {
        throw std::runtime_error("failed to get MNN output tensor");
    }

    // 输出可能位于非 CPU 后端内存。copyToHostTensor 后，业务侧才能稳定读取 float 数据。
    MNN::Tensor host_output(output, output->getDimensionType());
    output->copyToHostTensor(&host_output);

    const int output_size = host_output.elementSize();
    std::vector<float> logits(static_cast<size_t>(output_size));
    //为什么要拷贝？
    // - host_output.host<float>() 返回的指针指向 MNN 内部管理的内存，这块内存的生命周期和 MNN tensor 绑定。
    // - 如果后续 MNN 释放了这个 tensor 或者重用这块内存，host_output.host<float>() 就可能变成悬空指针。
    // - 通过 memcpy 把数据复制到业务侧管理的 vector 内存，确保分类结果在 classify 函数返回后依然有效。
    std::memcpy(logits.data(), host_output.host<float>(), logits.size() * sizeof(float));

    return softmax(logits);
}

int MNNClassifier::input_width() const {
    // 从模型输入 tensor 反查尺寸，使同一套代码可以迁移到其他分类模型。
    MNN::Tensor* input = interpreter_->getSessionInput(session_, nullptr);
    return input ? input->width() : 0;
}

int MNNClassifier::input_height() const {
    // MobileNetV2 当前是 224，但这里不写死，避免模型替换后预处理尺寸忘记同步。
    MNN::Tensor* input = interpreter_->getSessionInput(session_, nullptr);
    return input ? input->height() : 0;
}
