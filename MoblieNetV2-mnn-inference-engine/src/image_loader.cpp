#include "image_loader.h"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <stdexcept>

namespace {

// CoreFoundation / CoreGraphics 仍然使用 C 风格的所有权模型:
// - 函数名包含 Create / Copy 的对象，调用方拥有所有权。
// - 拥有所有权的对象必须在不再使用时调用对应的 Release。
//
// C++ 代码如果手写多处 CFRelease / CGImageRelease，很容易在中间 return/throw 时遗漏释放。
// ScopedCf 的作用就是把这些 C 对象包成 RAII:
// - 构造时接管对象。
// - 析构时自动释放。
// - 禁止拷贝，避免两个 wrapper 同时释放同一个底层对象。
//
// 它和 std::unique_ptr 的定位类似，只是 CoreFoundation/CoreGraphics 对象不是用 delete 释放，
// 而是要用 CFRelease、CGImageRelease、CGContextRelease 等不同函数释放。
template <typename T, void (*ReleaseFn)(T)>
class ScopedCf {
public:
    // 输入:
    // - value: Create/Copy 系列 API 返回的对象，可能为 nullptr。
    //
    // 工程目的:
    // - 把“谁负责释放资源”写进类型里。
    // - 后续代码只需要检查 get() 是否为空，不需要在每个异常分支手动释放。
    explicit ScopedCf(T value) : value_(value) {}

    ~ScopedCf() {
        if (value_) {
            ReleaseFn(value_);
        }
    }

    // 禁止拷贝是资源安全的关键。
    // 如果允许拷贝，两个 ScopedCf 实例会持有同一个 CF/CG 指针，
    // 析构时会 double release，表现可能是偶发 crash 或内存破坏。
    ScopedCf(const ScopedCf&) = delete;
    ScopedCf& operator=(const ScopedCf&) = delete;

    // 输出:
    // - 返回底层 C 指针，供 ImageIO/CoreGraphics API 使用。
    //
    // 注意:
    // - 调用者不拥有返回指针，不应该手动释放。
    // - 对象生命周期仍由当前 ScopedCf 实例管理。
    T get() const { return value_; }

private:
    T value_ = nullptr;
};

// CoreFoundation / CoreGraphics 对象使用不同的 release 函数签名。
// 这里分别包装成精确类型，避免在模板参数中依赖隐式函数指针转换。
void release_cf_string(CFStringRef value) {
    CFRelease(value);
}

void release_cf_url(CFURLRef value) {
    CFRelease(value);
}

void release_cg_image_source(CGImageSourceRef value) {
    CFRelease(value);
}

void release_cg_image(CGImageRef value) {
    CGImageRelease(value);
}

void release_cg_color_space(CGColorSpaceRef value) {
    CGColorSpaceRelease(value);
}

void release_cg_context(CGContextRef value) {
    CGContextRelease(value);
}

CFURLRef make_file_url(const std::string& path) {
    // 输入:
    // - path: C++ 层传入的 POSIX 文件路径，例如 assets/models/test_dog.jpg。
    //
    // 输出:
    // - 成功时返回一个新的 CFURLRef，调用方拥有所有权。
    // - 失败时返回 nullptr，由上层统一转成 C++ exception。
    //
    // 为什么不直接把 std::string 传给 ImageIO:
    // - ImageIO 是 Apple C API，读取本地文件时接受 CFURLRef。
    // - CFURLRef 又需要从 CFStringRef 构造。
    // - 这里把桥接细节集中封装，避免主流程被平台 API 噪声打断。
    ScopedCf<CFStringRef, release_cf_string> cf_path(
        CFStringCreateWithCString(nullptr, path.c_str(), kCFStringEncodingUTF8));
    if (!cf_path.get()) {
        return nullptr;
    }
    return CFURLCreateWithFileSystemPath(
        nullptr,
        cf_path.get(),
        kCFURLPOSIXPathStyle,
        false);
}

}  // namespace

RgbaImage load_rgba_image_resized(const std::string& image_path, int width, int height) {
    // 输入:
    // - image_path: JPEG/PNG 等 ImageIO 支持的图片路径。
    // - width/height: 目标模型输入尺寸，由 MNN input tensor 推导出来。
    //
    // 输出:
    // - RgbaImage: 已经 resize 到目标尺寸、像素连续存储的 RGBA byte buffer。
    //
    // 在推理 pipeline 中的位置:
    //   file/image source -> decode -> resize -> RGBA bytes
    //   -> MNN ImageProcess -> normalized RGB tensor -> MNN session
    //
    // 这里有意不做 mean/std 归一化:
    // - 归一化参数属于模型预处理契约，放在 MNNClassifier 更清晰。
    // - 图片加载层只负责把外部图片转换成统一尺寸/格式的中间表示。
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("target image size must be positive");
    }

    // Step 1: 把 C++ 文件路径转换成 ImageIO 可识别的 CFURL。
    //
    // url 是 ScopedCf 管理的对象。后续任何一步抛异常，url 都会自动释放。
    ScopedCf<CFURLRef, release_cf_url> url(make_file_url(image_path));
    if (!url.get()) {
        throw std::runtime_error("failed to create file URL: " + image_path);
    }

    // Step 2: 创建图片源。
    //
    // CGImageSource 可以理解成“图片文件容器解析器”:
    // - 它知道文件里有多少张图。
    // - 它负责识别 JPEG/PNG 等编码格式。
    // - 真正像素解码发生在后面的 CGImageSourceCreateImageAtIndex。
    ScopedCf<CGImageSourceRef, release_cg_image_source> source(
        CGImageSourceCreateWithURL(url.get(), nullptr));
    if (!source.get()) {
        throw std::runtime_error("failed to create image source: " + image_path);
    }

    // Step 3: 解码第 0 帧图片。
    //
    // 对普通 JPEG/PNG，第 0 帧就是整张图片。
    // 对 GIF/多页 TIFF 这类多帧格式，这个 demo 不处理动画或多页语义，只取第一帧。
    ScopedCf<CGImageRef, release_cg_image> image(
        CGImageSourceCreateImageAtIndex(source.get(), 0, nullptr));
    if (!image.get()) {
        throw std::runtime_error("failed to decode image: " + image_path);
    }

    // Step 4: 准备目标 RGBA buffer。
    //
    // result.pixels 是真正承载输出像素的内存。
    // 后面的 CGBitmapContextCreate 会把 CoreGraphics 的绘制目标绑定到这块 vector 内存，
    // 因此绘制完成后，不需要额外 memcpy，result.pixels 已经包含 resize 后的图像。
    RgbaImage result;
    result.width = width;
    result.height = height;
    result.pixels.resize(static_cast<size_t>(width * height * 4));

    // Step 5: 指定目标颜色空间。
    //
    // 使用 DeviceRGB 的原因:
    // - MNN MobileNetV2 的预处理按 RGB 三通道理解输入。
    // - CoreGraphics 在绘制时可以把源图的颜色表示转换到目标 RGB bitmap。
    // - 对 demo 和基线 benchmark 来说，这比手写颜色管理更可靠。
    ScopedCf<CGColorSpaceRef, release_cg_color_space> color_space(
        CGColorSpaceCreateDeviceRGB());
    if (!color_space.get()) {
        throw std::runtime_error("failed to create RGB color space");
    }

    // Step 6: 创建一个“画布”，画布内存直接指向 result.pixels。
    //
    // 关键参数:
    // - bitsPerComponent = 8: 每个 R/G/B/A 分量都是 uint8。
    // - bytesPerRow = width * 4: 每行 width 个像素，每个像素 4 字节。
    // - kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big:
    //   让内存布局稳定为 R,G,B,A 顺序，方便后续按 RGBA 喂给 MNN ImageProcess。
    //
    // 性能含义:
    // - 这条路径只分配一次目标 buffer。
    // - resize 直接绘制到目标 buffer，避免“先 decode 大图，再手动 resize，再复制”的多段内存搬运。
    // - 但 JPEG/PNG 解码和 CoreGraphics resize 仍然是 CPU 路径，不是硬件零拷贝最优路径。
    ScopedCf<CGContextRef, release_cg_context> context(
        CGBitmapContextCreate(
            result.pixels.data(),
            static_cast<size_t>(width),
            static_cast<size_t>(height),
            8,
            static_cast<size_t>(width * 4),
            color_space.get(),
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big));
    if (!context.get()) {
        throw std::runtime_error("failed to create bitmap context");
    }

    // Step 7: 解码后的 CGImage 被绘制到目标尺寸的 bitmap context。
    //
    // 这一步同时完成:
    // - 尺寸缩放: 原图尺寸 -> 模型输入尺寸。
    // - 像素格式落地: CoreGraphics 内部表示 -> result.pixels 的 RGBA 字节布局。
    //
    // 为什么这不是“绝对最优”:
    // - 它适合 macOS 桌面 demo 和可复现 benchmark。
    // - 如果目标是生产级端侧吞吐，应优先避免重复解码和重复 resize。
    // - Android 主线更可能走 CameraX YUV_420_888 -> NEON/libyuv -> RGB/float tensor，
    //   而不是先落地成 PNG/JPEG 文件再由 ImageIO 解码。
    CGContextDrawImage(
        context.get(),
        CGRectMake(0.0, 0.0, static_cast<double>(width), static_cast<double>(height)),
        image.get());

    return result;
}
