import Foundation

struct NativeInferenceBenchmarkRunner: InferenceBenchmarking {
    func runBenchmark(configuration: InferenceConfiguration, imagePath: String?) -> InferenceSnapshot {
        let configuration = configuration.normalized
        let bridge = NativeInferenceBridge()

        let requestedFramework: NativeInferenceFramework = configuration.framework == .mnn ? .mnn : .ncnn
        let requestedBackend: NativeInferenceBackend = {
            switch configuration.backend {
            case .cpu:
                return .cpu
            case .metal:
                return .metal
            case .vulkan:
                return .vulkan
            }
        }()

        let data: Data?
        do {
            data = try bridge.runBenchmark(for: requestedFramework, backend: requestedBackend, imagePath: imagePath)
        } catch {
            let fallback = DemoInferenceBenchmarkRunner().runBenchmark(configuration: configuration, imagePath: imagePath)
            var snapshot = fallback
            snapshot.notes = "Native runner failed: \(error.localizedDescription). " + snapshot.notes
            return snapshot
        }

        if let data,
           let payload = try? JSONDecoder().decode(BenchmarkSnapshotPayload.self, from: data) {
            return payload.toSnapshot()
        }

        let fallback = DemoInferenceBenchmarkRunner().runBenchmark(configuration: configuration, imagePath: imagePath)
        return fallback
    }
}
