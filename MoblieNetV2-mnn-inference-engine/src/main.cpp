#include "benchmark.h"
#include "image_loader.h"
#include "infer_utils.h"
#include "mnn_classifier.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    // 命令行参数保持显式路径，便于从仓库根目录或 build 目录分别运行。
    std::string model_path;
    std::string image_path;
    //assets/models/imagenet_classes.txt 的作用是把模型输出的类别编号翻译成人能看懂的类别名称，增强结果的可读性。
    //程序本身只能知道“第 258 类概率最高”，但不知道第 258 类是什么。assets/models/imagenet_classes.txt 按行保存类别名，行号就是类别 ID，所以程序读取后可以打印成：
    //class=258 prob=0.779527 label="Samoyed"
    std::string labels_path;

    // use_metal 只表示“请求 Metal”。如果 MNN 运行时打印 fallback，实际可能仍是 CPU。
    bool use_metal = false;

    // 线程数先固定在 demo 层，后续做性能表时可以扩展成命令行参数。
    int threads = 4;

    // warmup 与 runs 分开，避免首次运行开销污染稳定延迟。
    int warmup_runs = 10;
    int test_runs = 100;
};

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " <model.mnn> <image.jpg> [cpu|metal] [labels.txt] [warmup] [runs]\n";
}

Options parse_args(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        throw std::invalid_argument("missing required arguments");
    }

    Options options;
    options.model_path = argv[1];
    options.image_path = argv[2];

    if (argc >= 4) {
        const std::string backend = argv[3];
        // backend 参数刻意只接受 cpu/metal，避免拼写错误时悄悄落到 CPU。
        if (backend == "metal") {
            options.use_metal = true;
        } else if (backend != "cpu") {
            throw std::invalid_argument("backend must be cpu or metal");
        }
    }
    if (argc >= 5) {
        options.labels_path = argv[4];
    }
    if (argc >= 6) {
        options.warmup_runs = std::stoi(argv[5]);
    }
    if (argc >= 7) {
        options.test_runs = std::stoi(argv[6]);
    }
    return options;
}

std::vector<std::string> load_labels(const std::string& labels_path) {
    std::vector<std::string> labels;
    if (labels_path.empty()) {
        return labels;
    }

    std::ifstream input(labels_path);
    if (!input) {
        throw std::runtime_error("failed to open labels file: " + labels_path);
    }

    std::string line;
    while (std::getline(input, line)) {
        // labels 文件的行号就是 ImageNet class id，因此保持原顺序读入。
        labels.push_back(line);
    }
    return labels;
}

void print_topk_with_labels(
    const std::vector<float>& probs,
    const std::vector<std::string>& labels,
    int k) {
    std::cout << "Top-" << k << " result:\n";
    for (const TopKItem& item : topk(probs, k)) {
        std::cout << "  class=" << item.class_id
                  << " prob=" << item.probability;
        if (item.class_id >= 0 && item.class_id < static_cast<int>(labels.size())) {
            std::cout << " label=\"" << labels[static_cast<size_t>(item.class_id)] << "\"";
        }
        std::cout << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<std::string> labels = load_labels(options.labels_path);

        MNNClassifier classifier;
        const auto load_start = std::chrono::steady_clock::now();
        if (!classifier.load(options.model_path, options.threads, options.use_metal)) {
            return 1;
        }
        const auto load_end = std::chrono::steady_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

        // 输入尺寸来自 MNN tensor，而不是硬编码在图片加载器里。
        // 这让“模型文件决定预处理尺寸”的关系更清楚。
        const RgbaImage image = load_rgba_image_resized(
            options.image_path,
            classifier.input_width(),
            classifier.input_height());

        // 使用 lambda 包住一次完整推理，让 benchmark 不依赖 MNN 类型。
        // 后续替换 NCNN 或 Sherpa-ONNX 时，只要提供同样的一次执行函数即可复用统计逻辑。
        const BenchmarkResult result = run_benchmark(
            [&classifier, &image]() {
                return classifier.classify(image);
            },
            options.warmup_runs,
            options.test_runs);

        print_topk_with_labels(result.first_probs, labels, 5);
        std::cout << "[Load] model_load_ms=" << load_ms << "\n";
        std::cout << "Requested backend: "
                  << (options.use_metal ? "MNN Metal" : "MNN CPU") << "\n";
        std::cout << "Input: " << classifier.input_width()
                  << "x" << classifier.input_height() << "\n";
        std::cout << "Warmup: " << options.warmup_runs
                  << ", Runs: " << options.test_runs << "\n";
        std::cout << "Mean: " << result.mean << " ms\n";
        std::cout << "Median: " << result.median << " ms\n";
        std::cout << "P95: " << result.p95 << " ms\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
