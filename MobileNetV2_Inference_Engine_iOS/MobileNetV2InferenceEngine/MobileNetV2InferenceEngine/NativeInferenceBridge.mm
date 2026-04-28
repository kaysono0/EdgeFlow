#import "NativeInferenceBridge.h"

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#import <UIKit/UIKit.h>

#define MNN_METAL
#import <MNN/ImageProcess.hpp>
#import <MNN/Interpreter.hpp>
#import <MNN/MNNSharedContext.h>
#import <MNN/Tensor.hpp>

#include <ncnn/allocator.h>
#include <ncnn/gpu.h>
#include <ncnn/net.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct RankedItem {
    int index = -1;
    double probability = 0.0;
    std::string label;
};

struct BenchmarkPayload {
    std::string framework;
    std::string backend;
    std::string device;
    std::string model;
    std::string sourceImageName;
    std::string debugPreprocessedImagePath;
    std::string inputDescription;
    double loadMs = 0.0;
    int warmupRuns = 0;
    int benchmarkRuns = 0;
    double meanMs = 0.0;
    double medianMs = 0.0;
    double p95Ms = 0.0;
    RankedItem top1;
    std::vector<RankedItem> top5;
    std::string notes;
};

template <typename T>
static std::vector<T> softmax(const std::vector<T>& logits) {
    if (logits.empty()) {
        return {};
    }

    const auto max_it = std::max_element(logits.begin(), logits.end());
    const T max_value = *max_it;
    std::vector<T> exps;
    exps.reserve(logits.size());
    T sum = 0;
    for (T value : logits) {
        const T e = std::exp(value - max_value);
        exps.push_back(e);
        sum += e;
    }

    if (sum == 0) {
        return std::vector<T>(logits.size(), 0);
    }

    for (T& value : exps) {
        value /= sum;
    }
    return exps;
}

static std::vector<RankedItem> topk(const std::vector<double>& probabilities, const std::vector<std::string>& labels, int k) {
    std::vector<RankedItem> ranked;
    ranked.reserve(probabilities.size());
    for (size_t i = 0; i < probabilities.size(); ++i) {
        RankedItem item;
        item.index = static_cast<int>(i);
        item.probability = probabilities[i];
        if (i < labels.size()) {
            item.label = labels[i];
        }
        ranked.push_back(std::move(item));
    }

    if (k > static_cast<int>(ranked.size())) {
        k = static_cast<int>(ranked.size());
    }

    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(), [](const RankedItem& lhs, const RankedItem& rhs) {
        if (lhs.probability == rhs.probability) {
            return lhs.index < rhs.index;
        }
        return lhs.probability > rhs.probability;
    });
    ranked.resize(static_cast<size_t>(k));
    return ranked;
}

static NSData* jsonDataFromPayload(const BenchmarkPayload& payload, NSError** error) {
    NSMutableArray<NSDictionary<NSString*, id>*>* top5 = [NSMutableArray array];
    for (const auto& item : payload.top5) {
        [top5 addObject:@{
            @"index": @(item.index),
            @"label": [NSString stringWithUTF8String:item.label.c_str()],
            @"probability": @(item.probability),
        }];
    }

    NSDictionary<NSString*, id>* json = @{
        @"framework": [NSString stringWithUTF8String:payload.framework.c_str()],
        @"backend": [NSString stringWithUTF8String:payload.backend.c_str()],
        @"device": [NSString stringWithUTF8String:payload.device.c_str()],
        @"model": [NSString stringWithUTF8String:payload.model.c_str()],
        @"sourceImageName": [NSString stringWithUTF8String:payload.sourceImageName.c_str()],
        @"inputDescription": [NSString stringWithUTF8String:payload.inputDescription.c_str()],
        @"loadMs": @(payload.loadMs),
        @"warmupRuns": @(payload.warmupRuns),
        @"benchmarkRuns": @(payload.benchmarkRuns),
        @"meanMs": @(payload.meanMs),
        @"medianMs": @(payload.medianMs),
        @"p95Ms": @(payload.p95Ms),
        @"top1": @{
            @"index": @(payload.top1.index),
            @"label": [NSString stringWithUTF8String:payload.top1.label.c_str()],
            @"probability": @(payload.top1.probability),
        },
        @"top5": top5,
        @"notes": [NSString stringWithUTF8String:payload.notes.c_str()],
    };

    if (!payload.debugPreprocessedImagePath.empty()) {
        NSMutableDictionary<NSString*, id>* mutableJson = [json mutableCopy];
        mutableJson[@"debugPreprocessedImagePath"] = [NSString stringWithUTF8String:payload.debugPreprocessedImagePath.c_str()];
        json = [mutableJson copy];
    }

    if (![NSJSONSerialization isValidJSONObject:json]) {
        if (error) {
            *error = [NSError errorWithDomain:@"NativeInferenceBridge"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Generated payload is not valid JSON."}];
        }
        return nil;
    }

    NSData* data = [NSJSONSerialization dataWithJSONObject:json options:NSJSONWritingPrettyPrinted error:error];
    return data;
}

static std::string std_string_from_nsstring(NSString* string) {
    if (!string) {
        return {};
    }
    return std::string([string UTF8String] ?: "");
}

static std::string source_image_name_for_path(NSString* imagePath) {
    if (!imagePath) {
        return "samoyed.jpg";
    }
    NSString* name = [imagePath lastPathComponent];
    if (name.length == 0) {
        return "samoyed.jpg";
    }
    return std_string_from_nsstring(name);
}

static double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                         const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void benchmark_log(NSString* message) {
    NSLog(@"[NativeBenchmark] %@", message);
}

static void benchmark_log_timing(NSString* stage, double ms) {
    NSLog(@"[NativeBenchmark] %@: %.3f ms", stage, ms);
}

static NSString* resource_path(NSString* name, NSString* ext) {
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* path = [bundle pathForResource:name ofType:ext];
    if (path.length > 0) {
        return path;
    }
    path = [bundle pathForResource:name ofType:ext inDirectory:@"AppResources"];
    if (path.length > 0) {
        return path;
    }
    return nil;
}

static std::vector<uint8_t> load_rgba_image(NSString* imagePath, int targetWidth, int targetHeight) {
    NSURL* url = [NSURL fileURLWithPath:imagePath];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!source) {
        throw std::runtime_error("failed to create image source");
    }

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!image) {
        throw std::runtime_error("failed to decode image");
    }

    std::vector<uint8_t> rgba(static_cast<size_t>(targetWidth * targetHeight * 4));
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (!colorSpace) {
        CGImageRelease(image);
        throw std::runtime_error("failed to create device RGB colorspace");
    }

    CGContextRef context = CGBitmapContextCreate(
        rgba.data(),
        static_cast<size_t>(targetWidth),
        static_cast<size_t>(targetHeight),
        8,
        static_cast<size_t>(targetWidth * 4),
        colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    if (!context) {
        CGImageRelease(image);
        throw std::runtime_error("failed to create bitmap context");
    }

    CGContextDrawImage(context, CGRectMake(0, 0, targetWidth, targetHeight), image);
    CGContextRelease(context);
    CGImageRelease(image);
    return rgba;
}

static std::vector<uint8_t> load_rgba_image_resize_center_crop(NSString* imagePath,
                                                               int resizeShortSide,
                                                               int cropWidth,
                                                               int cropHeight) {
    NSURL* url = [NSURL fileURLWithPath:imagePath];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!source) {
        throw std::runtime_error("failed to create image source");
    }

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!image) {
        throw std::runtime_error("failed to decode image");
    }

    const size_t sourceWidth = CGImageGetWidth(image);
    const size_t sourceHeight = CGImageGetHeight(image);
    if (sourceWidth == 0 || sourceHeight == 0) {
        CGImageRelease(image);
        throw std::runtime_error("invalid source image size");
    }

    const double scale = static_cast<double>(resizeShortSide) /
        static_cast<double>(std::min(sourceWidth, sourceHeight));
    const size_t resizedWidth = static_cast<size_t>(std::llround(static_cast<double>(sourceWidth) * scale));
    const size_t resizedHeight = static_cast<size_t>(std::llround(static_cast<double>(sourceHeight) * scale));

    std::vector<uint8_t> resizedRGBA(resizedWidth * resizedHeight * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (!colorSpace) {
        CGImageRelease(image);
        throw std::runtime_error("failed to create device RGB colorspace");
    }

    CGContextRef resizeContext = CGBitmapContextCreate(
        resizedRGBA.data(),
        resizedWidth,
        resizedHeight,
        8,
        resizedWidth * 4,
        colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    if (!resizeContext) {
        CGColorSpaceRelease(colorSpace);
        CGImageRelease(image);
        throw std::runtime_error("failed to create resize bitmap context");
    }

    CGContextSetInterpolationQuality(resizeContext, kCGInterpolationHigh);
    CGContextDrawImage(resizeContext, CGRectMake(0, 0, resizedWidth, resizedHeight), image);
    CGImageRef resizedImage = CGBitmapContextCreateImage(resizeContext);
    CGContextRelease(resizeContext);
    CGColorSpaceRelease(colorSpace);
    CGImageRelease(image);
    if (!resizedImage) {
        throw std::runtime_error("failed to materialize resized image");
    }

    const size_t cropX = (resizedWidth > static_cast<size_t>(cropWidth))
        ? (resizedWidth - static_cast<size_t>(cropWidth)) / 2
        : 0;
    const size_t cropY = (resizedHeight > static_cast<size_t>(cropHeight))
        ? (resizedHeight - static_cast<size_t>(cropHeight)) / 2
        : 0;
    const size_t finalWidth = static_cast<size_t>(cropWidth);
    const size_t finalHeight = static_cast<size_t>(cropHeight);

    std::vector<uint8_t> rgba(finalWidth * finalHeight * 4);
    CGColorSpaceRef cropColorSpace = CGColorSpaceCreateDeviceRGB();
    if (!cropColorSpace) {
        CGImageRelease(resizedImage);
        throw std::runtime_error("failed to create crop colorspace");
    }

    CGContextRef cropContext = CGBitmapContextCreate(
        rgba.data(),
        finalWidth,
        finalHeight,
        8,
        finalWidth * 4,
        cropColorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(cropColorSpace);
    if (!cropContext) {
        CGImageRelease(resizedImage);
        throw std::runtime_error("failed to create crop bitmap context");
    }

    const CGRect cropRect = CGRectMake(cropX, cropY, finalWidth, finalHeight);
    CGImageRef croppedImage = CGImageCreateWithImageInRect(resizedImage, cropRect);
    CGImageRelease(resizedImage);
    if (!croppedImage) {
        CGContextRelease(cropContext);
        throw std::runtime_error("failed to create cropped image");
    }

    CGContextDrawImage(cropContext, CGRectMake(0, 0, finalWidth, finalHeight), croppedImage);
    CGContextRelease(cropContext);
    CGImageRelease(croppedImage);
    return rgba;
}

static NSString* write_rgba_image_to_temporary_png(const std::vector<uint8_t>& rgba,
                                                   int width,
                                                   int height,
                                                   NSString* fileName) {
    if (rgba.empty() || width <= 0 || height <= 0 || !fileName) {
        return nil;
    }

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (!colorSpace) {
        return nil;
    }

    CGDataProviderRef provider = CGDataProviderCreateWithData(
        nullptr,
        rgba.data(),
        rgba.size(),
        nullptr);
    if (!provider) {
        CGColorSpaceRelease(colorSpace);
        return nil;
    }

    CGImageRef image = CGImageCreate(
        static_cast<size_t>(width),
        static_cast<size_t>(height),
        8,
        32,
        static_cast<size_t>(width * 4),
        colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big,
        provider,
        nullptr,
        false,
        kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    if (!image) {
        return nil;
    }

    UIImage* uiImage = [UIImage imageWithCGImage:image];
    CGImageRelease(image);
    if (!uiImage) {
        return nil;
    }

    NSData* pngData = UIImagePNGRepresentation(uiImage);
    if (!pngData) {
        return nil;
    }

    NSString* filePath = [NSTemporaryDirectory() stringByAppendingPathComponent:fileName];
    NSURL* url = [NSURL fileURLWithPath:filePath];
    NSError* error = nil;
    if (![pngData writeToURL:url options:NSDataWritingAtomic error:&error]) {
        benchmark_log([NSString stringWithFormat:@"Failed to write preprocessed image: %@", error.localizedDescription ?: @"unknown error"]);
        return nil;
    }
    return filePath;
}

static void log_rgba_preview(NSString* prefix, const std::vector<uint8_t>& rgba, int width, int height) {
    const size_t pixelCount = rgba.size() / 4;
    const size_t sampleCount = std::min<size_t>(pixelCount, 4);
    NSMutableArray<NSString*>* samples = [NSMutableArray arrayWithCapacity:sampleCount];
    for (size_t i = 0; i < sampleCount; ++i) {
        const size_t base = i * 4;
        [samples addObject:[NSString stringWithFormat:@"#%zu=(%u,%u,%u,%u)",
                            i,
                            rgba[base + 0],
                            rgba[base + 1],
                            rgba[base + 2],
                            rgba[base + 3]]];
    }
    benchmark_log([NSString stringWithFormat:@"%@ width=%d height=%d pixels=%zu samples=[%@]",
                       prefix,
                       width,
                       height,
                       pixelCount,
                       [samples componentsJoinedByString:@", "]]);
}

static void log_float_tensor_summary(NSString* prefix, MNN::Tensor* tensor) {
    if (!tensor) {
        benchmark_log([NSString stringWithFormat:@"%@ tensor=nil", prefix]);
        return;
    }

    MNN::Tensor hostTensor(tensor, tensor->getDimensionType());
    tensor->copyToHostTensor(&hostTensor);

    const int count = hostTensor.elementSize();
    const float* data = hostTensor.host<float>();
    if (!data || count <= 0) {
        benchmark_log([NSString stringWithFormat:@"%@ elementCount=%d but no host data", prefix, count]);
        return;
    }

    float minValue = data[0];
    float maxValue = data[0];
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        const float value = data[i];
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
        sum += value;
    }

    NSMutableArray<NSString*>* samples = [NSMutableArray array];
    const int sampleCount = std::min(count, 6);
    for (int i = 0; i < sampleCount; ++i) {
        [samples addObject:[NSString stringWithFormat:@"%.5f", data[i]]];
    }

    benchmark_log([NSString stringWithFormat:@"%@ elementCount=%d min=%.5f max=%.5f mean=%.5f samples=[%@]",
                       prefix,
                       count,
                       minValue,
                       maxValue,
                       sum / static_cast<double>(count),
                       [samples componentsJoinedByString:@", "]]);
}

static void log_top_logits(NSString* prefix, const std::vector<float>& logits, int topK) {
    if (logits.empty() || topK <= 0) {
        benchmark_log([NSString stringWithFormat:@"%@ empty logits", prefix]);
        return;
    }

    struct RankedLogit {
        int index = -1;
        float value = 0.0f;
    };

    std::vector<RankedLogit> ranked;
    ranked.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        ranked.push_back(RankedLogit{static_cast<int>(i), logits[i]});
    }

    if (topK > static_cast<int>(ranked.size())) {
        topK = static_cast<int>(ranked.size());
    }

    std::partial_sort(ranked.begin(), ranked.begin() + topK, ranked.end(), [](const RankedLogit& lhs, const RankedLogit& rhs) {
        if (lhs.value == rhs.value) {
            return lhs.index < rhs.index;
        }
        return lhs.value > rhs.value;
    });

    NSMutableArray<NSString*>* samples = [NSMutableArray arrayWithCapacity:topK];
    for (int i = 0; i < topK; ++i) {
        [samples addObject:[NSString stringWithFormat:@"#%d=(idx=%d, logit=%.6f)",
                                                  i + 1,
                                                  ranked[static_cast<size_t>(i)].index,
                                                  ranked[static_cast<size_t>(i)].value]];
    }

    benchmark_log([NSString stringWithFormat:@"%@ %@", prefix, [samples componentsJoinedByString:@", "]]);
}

static std::vector<std::string> load_labels(NSString* labelsPath) {
    NSError* error = nil;
    NSString* contents = [NSString stringWithContentsOfFile:labelsPath encoding:NSUTF8StringEncoding error:&error];
    if (!contents) {
        throw std::runtime_error(std::string("failed to load labels: ") + std_string_from_nsstring(labelsPath));
    }

    NSArray<NSString*>* lines = [contents componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    std::vector<std::string> labels;
    labels.reserve(lines.count);
    for (NSString* line in lines) {
        labels.push_back(std_string_from_nsstring(line));
    }
    return labels;
}

static BenchmarkPayload make_payload(const std::string& framework,
                                     const std::string& backend,
                                     const std::string& model,
                                     const std::string& sourceImageName,
                                     const std::string& inputDescription,
                                     double loadMs,
                                     int warmupRuns,
                                     int benchmarkRuns,
                                     double meanMs,
                                     double medianMs,
                                     double p95Ms,
                                     const std::vector<RankedItem>& top,
                                     const std::string& notes) {
    BenchmarkPayload payload;
    payload.framework = framework;
    payload.backend = backend;
    payload.device = "iPhone";
    payload.model = model;
    payload.sourceImageName = sourceImageName;
    payload.inputDescription = inputDescription;
    payload.loadMs = loadMs;
    payload.warmupRuns = warmupRuns;
    payload.benchmarkRuns = benchmarkRuns;
    payload.meanMs = meanMs;
    payload.medianMs = medianMs;
    payload.p95Ms = p95Ms;
    if (!top.empty()) {
        payload.top1 = top.front();
        payload.top5 = top;
    }
    payload.notes = notes;
    return payload;
}

static BenchmarkPayload run_mnn_benchmark(NSString* imagePath, NativeInferenceBackend backend, NSError** error) {
    NSString* modelPath = resource_path(@"mobilenetv2", @"mnn");
    NSString* resolvedImagePath = imagePath;
    if (!resolvedImagePath || resolvedImagePath.length == 0) {
        resolvedImagePath = resource_path(@"samoyed", @"jpg");
        if (!resolvedImagePath || resolvedImagePath.length == 0) {
            resolvedImagePath = resource_path(@"test_dog", @"jpg");
        }
    }
    NSString* labelsPath = resource_path(@"imagenet_classes", @"txt");
    if (!modelPath || !resolvedImagePath || !labelsPath) {
        if (error) {
            *error = [NSError errorWithDomain:@"NativeInferenceBridge"
                                         code:2
                                     userInfo:@{NSLocalizedDescriptionKey: @"Missing MNN app resources."}];
        }
        return {};
    }

    const std::vector<std::string> labels = load_labels(labelsPath);

    const auto load_start = std::chrono::steady_clock::now();
    std::unique_ptr<MNN::Interpreter, void (*)(MNN::Interpreter*)> interpreter(
        MNN::Interpreter::createFromFile(modelPath.UTF8String), MNN::Interpreter::destroy);
    if (!interpreter) {
        throw std::runtime_error("failed to load MNN model");
    }

    MNN::ScheduleConfig config;
    config.type = backend == NativeInferenceBackendMetal ? MNN_FORWARD_METAL : MNN_FORWARD_CPU;
    config.numThread = 4;

    MNN::BackendConfig backendConfig;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    if (backend == NativeInferenceBackendMetal) {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Metal device is unavailable on this system");
        }
        queue = [device newCommandQueue];
        MNNMetalSharedContext context;
        context.device = device;
        context.queue = queue;
        backendConfig.sharedContext = &context;
        config.backendConfig = &backendConfig;
    }

    std::shared_ptr<MNN::Session> session(interpreter->createSession(config), [interpreter = interpreter.get()](MNN::Session* s) {
        if (interpreter && s) {
            interpreter->releaseSession(s);
        }
    });
    if (!session) {
        throw std::runtime_error("failed to create MNN session");
    }

    MNN::Tensor* input = interpreter->getSessionInput(session.get(), nullptr);
    if (!input) {
        throw std::runtime_error("failed to obtain MNN input tensor");
    }

    const int inputWidth = input->width();
    const int inputHeight = input->height();
    const std::vector<uint8_t> rgba = load_rgba_image_resize_center_crop(resolvedImagePath, 256, inputWidth, inputHeight);
    benchmark_log([NSString stringWithFormat:
                       @"MNN source image resolved: %@ -> %@",
                       imagePath ?: @"<default>",
                       resolvedImagePath]);
    benchmark_log([NSString stringWithFormat:
                       @"MNN preprocess recipe: resize_short_side=256 center_crop=%dx%d",
                       inputWidth,
                       inputHeight]);
    log_rgba_preview(@"MNN source RGBA preview", rgba, inputWidth, inputHeight);

    const float mean[3] = {123.675f, 116.28f, 103.53f};
    const float normal[3] = {1.0f / 58.395f, 1.0f / 57.12f, 1.0f / 57.375f};

    MNN::CV::ImageProcess::Config processConfig;
    processConfig.sourceFormat = MNN::CV::RGBA;
    processConfig.destFormat = MNN::CV::RGB;
    ::memcpy(processConfig.mean, mean, sizeof(mean));
    ::memcpy(processConfig.normal, normal, sizeof(normal));

    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(processConfig));
    if (!process) {
        throw std::runtime_error("failed to create MNN image process");
    }

    benchmark_log([NSString stringWithFormat:
                       @"MNN input tensor shape: w=%d h=%d c=%d type=%d dimType=%d",
                       input->width(),
                       input->height(),
                       input->channel(),
                       input->getType(),
                       input->getDimensionType()]);
    log_float_tensor_summary(@"MNN input tensor summary", input);

    const auto load_end = std::chrono::steady_clock::now();
    const double loadMs = std::chrono::duration<double, std::milli>(load_end - load_start).count();

    const int warmupRuns = 5;
    const int benchmarkRuns = 20;
    for (int i = 0; i < warmupRuns; ++i) {
        MNN::Tensor* runInput = interpreter->getSessionInput(session.get(), nullptr);
        if (!runInput) {
            throw std::runtime_error("failed to obtain MNN input tensor");
        }
        process->convert(rgba.data(), inputWidth, inputHeight, inputWidth * 4, runInput);
        interpreter->runSession(session.get());
    }

    std::vector<double> latencies;
    latencies.reserve(static_cast<size_t>(benchmarkRuns));
    std::vector<float> logits;

    for (int i = 0; i < benchmarkRuns; ++i) {
        const auto start = std::chrono::steady_clock::now();
        MNN::Tensor* runInput = interpreter->getSessionInput(session.get(), nullptr);
        if (!runInput) {
            throw std::runtime_error("failed to obtain MNN input tensor");
        }
        process->convert(rgba.data(), inputWidth, inputHeight, inputWidth * 4, runInput);
        interpreter->runSession(session.get());
        const auto end = std::chrono::steady_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    MNN::Tensor* output = interpreter->getSessionOutput(session.get(), nullptr);
    if (!output) {
        throw std::runtime_error("failed to obtain MNN output tensor");
    }

    MNN::Tensor hostOutput(output, output->getDimensionType());
    output->copyToHostTensor(&hostOutput);
    const int outputSize = hostOutput.elementSize();
    logits.resize(static_cast<size_t>(outputSize));
    ::memcpy(logits.data(), hostOutput.host<float>(), sizeof(float) * static_cast<size_t>(outputSize));
    benchmark_log([NSString stringWithFormat:
                       @"MNN output tensor shape: w=%d h=%d c=%d elements=%d type=%d dimType=%d",
                       output->width(),
                       output->height(),
                       output->channel(),
                       output->width() * output->height() * output->channel(),
                       output->getType(),
                       output->getDimensionType()]);
    log_top_logits(@"MNN raw output top5 logits:", logits, 5);

    std::vector<double> probabilities;
    {
        std::vector<double> logitsAsDouble;
        logitsAsDouble.reserve(logits.size());
        for (float value : logits) {
            logitsAsDouble.push_back(static_cast<double>(value));
        }
        probabilities = softmax(logitsAsDouble);
    }

    std::vector<RankedItem> top = topk(probabilities, labels, 5);
    const std::string sourceImageName = source_image_name_for_path(resolvedImagePath);
    std::string debugPreprocessedImagePath;
    if (backend == NativeInferenceBackendCPU || backend == NativeInferenceBackendMetal) {
        NSString* fileName = [NSString stringWithFormat:@"mnn-preprocessed-%@-%@.png",
                                                        [NSString stringWithUTF8String:sourceImageName.c_str()],
                                                        [NSUUID UUID].UUIDString];
        NSString* debugPath = write_rgba_image_to_temporary_png(rgba, inputWidth, inputHeight, fileName);
        if (debugPath.length > 0) {
            debugPreprocessedImagePath = std_string_from_nsstring(debugPath);
            benchmark_log([NSString stringWithFormat:@"MNN preprocessed image exported: %@", debugPath]);
        }
    }
    std::vector<double> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    const double meanMs = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    const double medianMs = sorted[sorted.size() / 2];
    const size_t p95Index = static_cast<size_t>(std::ceil(sorted.size() * 0.95)) - 1;
    const double p95Ms = sorted[std::min(p95Index, sorted.size() - 1)];

    std::string backendNote = backend == NativeInferenceBackendMetal
        ? "Requested Metal backend. MNN Metal path enabled via shared context."
        : "Requested CPU backend.";
    BenchmarkPayload payload = make_payload(
        "MNN",
        backend == NativeInferenceBackendMetal ? "Metal" : "CPU",
        "MobileNetV2",
        sourceImageName,
        "1x3x224x224",
        loadMs,
        warmupRuns,
        benchmarkRuns,
        meanMs,
        medianMs,
        p95Ms,
        top,
        backendNote);
    payload.debugPreprocessedImagePath = debugPreprocessedImagePath;
    return payload;
}

static BenchmarkPayload run_ncnn_benchmark(NSString* imagePath, NativeInferenceBackend backend, NSError** error) {
    if (backend == NativeInferenceBackendMetal) {
        if (error) {
            *error = [NSError errorWithDomain:@"NativeInferenceBridge"
                                         code:3
                                     userInfo:@{NSLocalizedDescriptionKey: @"NCNN Metal is unavailable on iOS in this app. Use CPU or switch to Vulkan."}];
        }
        return {};
    }

    const bool requestedVulkan = backend == NativeInferenceBackendVulkan;
    bool useVulkan = requestedVulkan;
    bool fallbackToCPU = false;

#if !NCNN_VULKAN
    if (requestedVulkan) {
        useVulkan = false;
        fallbackToCPU = true;
    }
#endif

    NSString* paramPath = resource_path(@"mobilenetv2", @"param");
    NSString* binPath = resource_path(@"mobilenetv2", @"bin");
    NSString* resolvedImagePath = imagePath;
    if (!resolvedImagePath || resolvedImagePath.length == 0) {
        resolvedImagePath = resource_path(@"samoyed", @"jpg");
        if (!resolvedImagePath || resolvedImagePath.length == 0) {
            resolvedImagePath = resource_path(@"test_dog", @"jpg");
        }
    }
    NSString* labelsPath = resource_path(@"imagenet_classes", @"txt");
    if (!paramPath || !binPath || !resolvedImagePath || !labelsPath) {
        if (error) {
            *error = [NSError errorWithDomain:@"NativeInferenceBridge"
                                         code:4
                                     userInfo:@{NSLocalizedDescriptionKey: @"Missing NCNN app resources."}];
        }
        return {};
    }

    const std::vector<std::string> labels = load_labels(labelsPath);
    const std::string param = std_string_from_nsstring(paramPath);
    const std::string bin = std_string_from_nsstring(binPath);
    const std::vector<uint8_t> rgba = load_rgba_image(resolvedImagePath, 224, 224);

    const auto load_start = std::chrono::steady_clock::now();
    ncnn::Net net;
    net.opt.use_vulkan_compute = useVulkan;
    net.opt.use_packing_layout = true;
    net.opt.num_threads = 4;

#if NCNN_VULKAN
    ncnn::VulkanDevice* vkdev = nullptr;
    std::unique_ptr<ncnn::VkBlobAllocator> blobVkAllocator;
    std::unique_ptr<ncnn::VkStagingAllocator> stagingVkAllocator;
    if (useVulkan) {
        vkdev = ncnn::get_gpu_device(0);
        if (!vkdev || !vkdev->is_valid()) {
            benchmark_log([NSString stringWithFormat:
                               @"NCNN Vulkan device unavailable, falling back to CPU. gpuCount=%d defaultGpuIndex=%d",
                               ncnn::get_gpu_count(),
                               ncnn::get_default_gpu_index()]);
            useVulkan = false;
            fallbackToCPU = true;
            net.opt.use_vulkan_compute = false;
        } else {
            benchmark_log([NSString stringWithFormat:
                               @"NCNN Vulkan device selected: %s, type=%d, vendor=0x%04x, roughScore=%u, computeQueues=%u",
                               vkdev->info.device_name() ?: "",
                               vkdev->info.type(),
                               vkdev->info.vendor_id(),
                               vkdev->info.rough_score(),
                               vkdev->info.compute_queue_count()]);
            blobVkAllocator = std::make_unique<ncnn::VkBlobAllocator>(vkdev);
            stagingVkAllocator = std::make_unique<ncnn::VkStagingAllocator>(vkdev);
            net.opt.blob_vkallocator = blobVkAllocator.get();
            net.opt.workspace_vkallocator = blobVkAllocator.get();
            net.opt.staging_vkallocator = stagingVkAllocator.get();
            net.set_vulkan_device(vkdev);
        }
    }
#endif

    benchmark_log([NSString stringWithFormat:
                       @"NCNN backend state: requestedVulkan=%d enabledVulkan=%d gpuCount=%d defaultGpuIndex=%d numThreads=%d packingLayout=%d",
                       requestedVulkan ? 1 : 0,
                       net.opt.use_vulkan_compute ? 1 : 0,
                       ncnn::get_gpu_count(),
                       ncnn::get_default_gpu_index(),
                       net.opt.num_threads,
                       net.opt.use_packing_layout ? 1 : 0]);

    const auto load_param_start = std::chrono::steady_clock::now();
    if (net.load_param(param.c_str())) {
        throw std::runtime_error("failed to load NCNN param");
    }
    const auto load_param_end = std::chrono::steady_clock::now();
    benchmark_log_timing(@"NCNN load_param", elapsed_ms(load_param_start, load_param_end));

    const auto load_model_start = std::chrono::steady_clock::now();
    if (net.load_model(bin.c_str())) {
        throw std::runtime_error("failed to load NCNN bin");
    }
    const auto load_model_end = std::chrono::steady_clock::now();
    benchmark_log_timing(@"NCNN load_model", elapsed_ms(load_model_start, load_model_end));
    const auto load_end = load_model_end;
    const double loadMs = elapsed_ms(load_start, load_end);

    const auto preprocess_start = std::chrono::steady_clock::now();
    ncnn::Mat input = ncnn::Mat::from_pixels(rgba.data(), ncnn::Mat::PIXEL_RGBA2RGB, 224, 224);
    const float meanVals[3] = {123.675f, 116.28f, 103.53f};
    const float normVals[3] = {1.0f / 58.395f, 1.0f / 57.12f, 1.0f / 57.375f};
    input.substract_mean_normalize(meanVals, normVals);
    const auto preprocess_end = std::chrono::steady_clock::now();
    benchmark_log_timing(@"NCNN preprocess (from_pixels + normalize)", elapsed_ms(preprocess_start, preprocess_end));

    const int warmupRuns = 5;
    const int benchmarkRuns = 20;
    for (int i = 0; i < warmupRuns; ++i) {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", input);
        ncnn::Mat output;
        ex.extract("out0", output);
    }

    std::vector<double> latencies;
    latencies.reserve(static_cast<size_t>(benchmarkRuns));
    ncnn::Mat lastOutput;
    double createExtractorTotalMs = 0.0;
    double inputTotalMs = 0.0;
    double extractTotalMs = 0.0;
    for (int i = 0; i < benchmarkRuns; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const auto create_extractor_start = std::chrono::steady_clock::now();
        ncnn::Extractor ex = net.create_extractor();
        const auto create_extractor_end = std::chrono::steady_clock::now();
        createExtractorTotalMs += elapsed_ms(create_extractor_start, create_extractor_end);

        const auto input_start = std::chrono::steady_clock::now();
        ex.input("in0", input);
        const auto input_end = std::chrono::steady_clock::now();
        inputTotalMs += elapsed_ms(input_start, input_end);

        const auto extract_start = std::chrono::steady_clock::now();
        ex.extract("out0", lastOutput);
        const auto extract_end = std::chrono::steady_clock::now();
        extractTotalMs += elapsed_ms(extract_start, extract_end);

        const auto end = extract_end;
        latencies.push_back(elapsed_ms(start, end));

        if (i == 0) {
            benchmark_log_timing(@"NCNN first run create_extractor", elapsed_ms(create_extractor_start, create_extractor_end));
            benchmark_log_timing(@"NCNN first run input", elapsed_ms(input_start, input_end));
            benchmark_log_timing(@"NCNN first run extract", elapsed_ms(extract_start, extract_end));
            benchmark_log_timing(@"NCNN first run total", elapsed_ms(start, end));
        } else if (i < 3) {
            benchmark_log_timing([NSString stringWithFormat:@"NCNN run %d create_extractor", i + 1], elapsed_ms(create_extractor_start, create_extractor_end));
            benchmark_log_timing([NSString stringWithFormat:@"NCNN run %d input", i + 1], elapsed_ms(input_start, input_end));
            benchmark_log_timing([NSString stringWithFormat:@"NCNN run %d extract", i + 1], elapsed_ms(extract_start, extract_end));
            benchmark_log_timing([NSString stringWithFormat:@"NCNN run %d total", i + 1], elapsed_ms(start, end));
        }
    }

    benchmark_log_timing(@"NCNN avg create_extractor", createExtractorTotalMs / static_cast<double>(benchmarkRuns));
    benchmark_log_timing(@"NCNN avg input (host->GPU upload on Vulkan)", inputTotalMs / static_cast<double>(benchmarkRuns));
    benchmark_log_timing(@"NCNN avg extract (compute + GPU->host download on Vulkan)", extractTotalMs / static_cast<double>(benchmarkRuns));
    benchmark_log([NSString stringWithFormat:
                       @"NCNN output shape: w=%d h=%d c=%d elements=%d",
                       lastOutput.w,
                       lastOutput.h,
                       lastOutput.c,
                       lastOutput.w * lastOutput.h * lastOutput.c]);

    lastOutput = lastOutput.reshape(lastOutput.w * lastOutput.h * lastOutput.c);
    std::vector<double> logits;
    logits.reserve(static_cast<size_t>(lastOutput.w));
    for (int i = 0; i < lastOutput.w; ++i) {
        logits.push_back(static_cast<double>(lastOutput[i]));
    }

    const std::vector<double> probabilities = softmax(logits);
    std::vector<RankedItem> top = topk(probabilities, labels, 5);

    std::vector<double> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    const double meanMs = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    const double medianMs = sorted[sorted.size() / 2];
    const size_t p95Index = static_cast<size_t>(std::ceil(sorted.size() * 0.95)) - 1;
    const double p95Ms = sorted[std::min(p95Index, sorted.size() - 1)];

    const std::string sourceImageName = source_image_name_for_path(resolvedImagePath);
    const std::string backendName = useVulkan ? "Vulkan" : "CPU";
    const std::string backendNote = requestedVulkan
        ? (fallbackToCPU
               ? "Requested Vulkan backend, but no valid Vulkan device was available. NCNN CPU path is used."
               : "Requested Vulkan backend. NCNN Vulkan / MoltenVK path is used.")
        : "Requested CPU backend. NCNN CPU path is used.";

    return make_payload(
        "NCNN",
        backendName,
        "MobileNetV2",
        sourceImageName,
        "1x3x224x224",
        loadMs,
        warmupRuns,
        benchmarkRuns,
        meanMs,
        medianMs,
        p95Ms,
        top,
        backendNote);
}

} // namespace

@implementation NativeInferenceBridge

- (NSData * _Nullable)runBenchmarkForFramework:(NativeInferenceFramework)framework
                                       backend:(NativeInferenceBackend)backend
                                     imagePath:(NSString * _Nullable)imagePath
                                         error:(NSError * _Nullable * _Nullable)error {
    @try {
        try {
            BenchmarkPayload payload;
            if (framework == NativeInferenceFrameworkMNN) {
                payload = run_mnn_benchmark(imagePath, backend, error);
            } else {
                payload = run_ncnn_benchmark(imagePath, backend, error);
            }

            if (payload.framework.empty()) {
                return nil;
            }
            return jsonDataFromPayload(payload, error);
        } catch (const std::exception& exception) {
            if (error) {
                *error = [NSError errorWithDomain:@"NativeInferenceBridge"
                                             code:998
                                         userInfo:@{
                                             NSLocalizedDescriptionKey: [NSString stringWithUTF8String:exception.what()]
                                         }];
            }
            return nil;
        }
    } @catch (NSException* exception) {
        if (error) {
            *error = [NSError errorWithDomain:@"NativeInferenceBridge"
                                         code:999
                                     userInfo:@{
                                         NSLocalizedDescriptionKey: exception.reason ?: @"Native inference failed."
                                     }];
        }
        return nil;
    }
}

@end
