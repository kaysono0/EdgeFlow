import Combine
import Foundation
import Photos

protocol InferenceBenchmarking: Sendable {
    func runBenchmark(configuration: InferenceConfiguration, imagePath: String?) -> InferenceSnapshot
}

struct DemoInferenceBenchmarkRunner: InferenceBenchmarking {
    func runBenchmark(configuration: InferenceConfiguration, imagePath: String?) -> InferenceSnapshot {
        let configuration = configuration.normalized
        Thread.sleep(forTimeInterval: 0.35)
        let latencyJitter: Double = configuration.framework == .ncnn ? 0.12 : 0.0
        let backendJitter: Double = configuration.backend == .cpu ? 0.32 : 0.0
        let loadMilliseconds = configuration.framework == .ncnn ? 2.062 : 2.418
        let meanMilliseconds = 5.204 + latencyJitter + backendJitter
        let medianMilliseconds = 5.081 + latencyJitter + backendJitter
        let p95Milliseconds = 5.432 + latencyJitter + backendJitter
        let top1Probability = configuration.framework == .ncnn ? 0.77511 : 0.78123
        let sourceImageName = imagePath.flatMap { URL(fileURLWithPath: $0).lastPathComponent } ?? "samoyed.jpg"
        return InferenceSnapshot(
            framework: configuration.framework.rawValue,
            backend: configuration.backend.rawValue,
            device: "iPhone",
            modelName: "MobileNetV2",
            sourceImageName: sourceImageName,
            loadMilliseconds: loadMilliseconds,
            warmupRuns: 5,
            benchmarkRuns: 20,
            meanMilliseconds: meanMilliseconds,
            medianMilliseconds: medianMilliseconds,
            p95Milliseconds: p95Milliseconds,
            top1Index: 258,
            top1Label: "Samoyed",
            top1Probability: top1Probability,
            top5: [
                (258, "Samoyed", top1Probability),
                (259, "Pomeranian", 0.09211),
                (257, "Eskimo dog", 0.05242),
                (261, "keeshond", 0.04120),
                (270, "malamute", 0.03104),
            ],
            notes: "\(configuration.runnerNotes) Demo payload until the C++ bridge is wired.",
            inputDescription: "1x3x224x224"
        )
    }
}

@MainActor
final class InferenceViewModel: ObservableObject {
    @Published var snapshot: InferenceSnapshot
    @Published var report: BenchmarkReportPayload
    @Published var isRunning = false
    @Published var statusText = "Ready"
    @Published var runToken = UUID()
    @Published var loadedSource = "Demo payload"
    @Published var selectedConfiguration = InferenceConfiguration()
    @Published var selectedTestImagePath: String?
    @Published var selectedTestImageName = "samoyed.jpg"

    private let runner: InferenceBenchmarking

    init(runner: InferenceBenchmarking? = nil) {
        let actualRunner = runner ?? NativeInferenceBenchmarkRunner()
        let initialConfiguration = InferenceConfiguration().normalized
        self.runner = actualRunner
        self.selectedConfiguration = initialConfiguration
        let initialImagePath = Self.defaultTestImagePath()
        self.selectedTestImagePath = initialImagePath
        self.selectedTestImageName = initialImagePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "samoyed.jpg"
        let initialSnapshot = Self.makeInitialSnapshot(configuration: initialConfiguration)
        self.snapshot = initialSnapshot
        self.report = ReportDocumentBuilder.makeReport(from: initialSnapshot)
        self.statusText = "Ready"
        self.loadedSource = self.selectedTestImageName
    }

    func importSnapshot(from data: Data, sourceName: String) {
        do {
            if let importedReport = try? BenchmarkSnapshotImporter.importReport(from: data),
               let importedSnapshot = importedReport.primarySnapshot() {
                snapshot = importedSnapshot
                report = importedReport
            } else {
                let imported = try BenchmarkSnapshotImporter.importSnapshot(from: data)
                snapshot = imported
                report = ReportDocumentBuilder.makeReport(from: imported)
            }
            statusText = "Imported \(sourceName)"
            loadedSource = sourceName
            runToken = UUID()
        } catch {
            statusText = "Failed to import \(sourceName)"
            loadedSource = "Import failed"
        }
    }

    func configurationDidChange() {
        let normalized = selectedConfiguration.normalized
        if normalized != selectedConfiguration {
            selectedConfiguration = normalized
        }
        statusText = "Selected \(selectedConfiguration.displayName)"
        loadedSource = selectedTestImageName
    }

    func selectTestImage(at path: String, sourceName: String) {
        selectedTestImagePath = path
        selectedTestImageName = sourceName
        loadedSource = sourceName
        statusText = "Selected test image: \(sourceName)"
    }

    func selectFramework(_ framework: InferenceFramework) {
        selectedConfiguration.framework = framework
        configurationDidChange()
    }

    func selectBackend(_ backend: InferenceBackend) {
        selectedConfiguration.backend = backend
        configurationDidChange()
    }

    func run() {
        let configuration = selectedConfiguration.normalized
        if configuration != selectedConfiguration {
            selectedConfiguration = configuration
        }
        isRunning = true
        statusText = "Running benchmark for \(configuration.displayName)..."
        let runner = runner
        let imagePath = selectedTestImagePath ?? Self.defaultTestImagePath()
        DispatchQueue.global(qos: .userInitiated).async {
            let next = runner.runBenchmark(configuration: configuration, imagePath: imagePath)
            DispatchQueue.main.async {
                self.snapshot = next
                self.report = ReportDocumentBuilder.makeReport(from: next)
                self.statusText = "Benchmark completed for \(configuration.displayName)"
                self.loadedSource = next.sourceImageName
                self.runToken = UUID()
                self.isRunning = false
            }
        }
    }

    func savePreprocessedImageToPhotos() {
        guard let imagePath = snapshot.debugPreprocessedImagePath, !imagePath.isEmpty else {
            statusText = "No preprocessed image available to save"
            return
        }

        let fileURL = URL(fileURLWithPath: imagePath)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            statusText = "Preprocessed image file missing"
            return
        }

        statusText = "Requesting Photos permission..."
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                Task { @MainActor in
                    self.statusText = "Photos permission denied"
                }
                return
            }

            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAssetFromImage(atFileURL: fileURL)
            }) { success, error in
                Task { @MainActor in
                    if success {
                        self.statusText = "Saved preprocessed image to Photos"
                    } else {
                        let reason = error?.localizedDescription ?? "unknown error"
                        self.statusText = "Failed to save preprocessed image: \(reason)"
                    }
                }
            }
        }
    }

    private static func makeInitialSnapshot(configuration: InferenceConfiguration) -> InferenceSnapshot {
        InferenceSnapshot(
            framework: configuration.framework.rawValue,
            backend: configuration.backend.rawValue,
            device: "iPhone",
            modelName: "MobileNetV2",
            sourceImageName: "samoyed.jpg",
            loadMilliseconds: 0.0,
            warmupRuns: 0,
            benchmarkRuns: 0,
            meanMilliseconds: 0.0,
            medianMilliseconds: 0.0,
            p95Milliseconds: 0.0,
            top1Index: 258,
            top1Label: "Samoyed",
            top1Probability: 0.0,
            top5: [
                (258, "Samoyed", 0.0),
                (259, "Pomeranian", 0.0),
                (257, "Eskimo dog", 0.0),
                (261, "keeshond", 0.0),
                (270, "malamute", 0.0),
            ],
            notes: "Tap Run to start native inference.",
            inputDescription: "1x3x224x224"
        )
    }

    private static func defaultTestImagePath() -> String? {
        let bundle = Bundle.main
        return bundle.path(forResource: "samoyed", ofType: "jpg")
        ?? bundle.path(forResource: "samoyed", ofType: "jpg", inDirectory: "AppResources")
        ?? bundle.path(forResource: "test_dog", ofType: "jpg")
        ?? bundle.path(forResource: "test_dog", ofType: "jpg", inDirectory: "AppResources")
    }
}
