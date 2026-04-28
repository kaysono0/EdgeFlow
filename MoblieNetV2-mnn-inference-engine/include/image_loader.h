#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct RgbaImage {
    // 当前 demo 在图片加载阶段就完成 resize，因此这里的尺寸应直接等于模型输入尺寸。
    //
    // 这相当于把真实端侧 pipeline 里的“输入适配层”提前做掉:
    // - 图片文件/相机帧通常不是模型输入尺寸。
    // - 推理引擎通常只关心固定形状 tensor。
    // - 因此这里返回的 width/height 是后续 MNN ImageProcess 的契约边界。
    int width = 0;
    int height = 0;

    // 像素按 RGBA/RGBA/RGBA... 连续存储，每个像素 4 字节。
    // MNN ImageProcess 后续会把它转换成模型需要的 RGB float tensor。
    //
    // 为什么这里用 RGBA，而不是直接输出 RGB float:
    // - macOS ImageIO/CoreGraphics 对 32-bit RGBA bitmap 的路径最稳定。
    // - MNN 自带 ImageProcess 已经负责颜色通道转换、mean/normal 归一化和写入 tensor。
    // - 这样图片解码层只负责“得到可控尺寸的字节图”，不和模型预处理参数强耦合。
    std::vector<uint8_t> pixels;
};

// 解码图片并缩放到指定尺寸。
//
// 输入:
// - image_path: JPEG/PNG 等 ImageIO 支持的图片路径。
// - width/height: 目标尺寸，通常来自 MNN 输入 tensor，例如 MobileNetV2 的 224x224。
//
// 输出:
// - RgbaImage，像素格式固定为 RGBA，方便后续统一送入 MNN ImageProcess。
//
// 工程目的:
// - 第三周目标是推理引擎闭环，不把样例成败绑定到 OpenCV 是否安装。
// - Android 主线以后会从 CameraX YUV 输入；这里先在 macOS 用 ImageIO 形成等价的“图像输入端”。
// - 这个函数不是最终 Android 高性能预处理方案，而是 macOS/MNN 基线里的可复现输入路径。
//
// 性能边界:
// - 对单张图片 benchmark，JPEG/PNG 解码通常不是主测对象，主要看 MNN session 推理耗时。
// - 若要测端到端延迟，可以把本函数也纳入计时。
// - 若要追求更高吞吐，应考虑缓存已解码/已 resize 的像素，或使用平台硬件路径/零拷贝输入。
RgbaImage load_rgba_image_resized(const std::string& image_path, int width, int height);
