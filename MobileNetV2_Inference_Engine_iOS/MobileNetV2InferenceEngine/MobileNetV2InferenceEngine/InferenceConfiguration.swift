import Foundation

enum InferenceFramework: String, CaseIterable, Identifiable {
    case mnn = "MNN"
    case ncnn = "NCNN"

    var id: String { rawValue }
}

enum InferenceBackend: String, CaseIterable, Identifiable {
    case cpu = "CPU"
    case metal = "Metal"
    case vulkan = "Vulkan"

    var id: String { rawValue }
}

struct InferenceConfiguration: Equatable {
    var framework: InferenceFramework = .mnn
    var backend: InferenceBackend = .metal

    var displayName: String {
        "\(framework.rawValue) / \(backend.rawValue)"
    }

    var availableBackends: [InferenceBackend] {
        switch framework {
        case .mnn:
            return [.cpu, .metal]
        case .ncnn:
            return [.cpu, .vulkan]
        }
    }

    var preferredBackend: InferenceBackend {
        availableBackends.contains(backend) ? backend : (framework == .mnn ? .metal : .vulkan)
    }

    var normalized: InferenceConfiguration {
        var copy = self
        copy.backend = preferredBackend
        return copy
    }

    var runnerNotes: String {
        switch (framework, backend) {
        case (.mnn, .metal):
            return "MNN Metal is the native Apple GPU path."
        case (.mnn, .cpu):
            return "MNN CPU is the baseline path."
        case (.ncnn, .cpu):
            return "NCNN CPU is the baseline path."
        case (.ncnn, .vulkan):
            return "NCNN Vulkan is the native Apple GPU path via MoltenVK."
        default:
            return "Selected backend is unavailable for this framework."
        }
    }
}
