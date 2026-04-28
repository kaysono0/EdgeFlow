import Foundation

struct BenchmarkSnapshotPayload: Codable {
    struct RankedItem: Codable {
        let index: Int
        let label: String
        let probability: Double
    }

    let framework: String
    let backend: String
    let device: String
    let model: String
    let sourceImageName: String?
    let debugPreprocessedImagePath: String?
    let inputDescription: String
    let loadMs: Double
    let warmupRuns: Int
    let benchmarkRuns: Int
    let meanMs: Double
    let medianMs: Double
    let p95Ms: Double
    let top1: RankedItem
    let top5: [RankedItem]
    let notes: String

    func toSnapshot() -> InferenceSnapshot {
        InferenceSnapshot(
            framework: framework,
            backend: backend,
            device: device,
            modelName: model,
            sourceImageName: sourceImageName ?? "samoyed.jpg",
            debugPreprocessedImagePath: debugPreprocessedImagePath,
            loadMilliseconds: loadMs,
            warmupRuns: warmupRuns,
            benchmarkRuns: benchmarkRuns,
            meanMilliseconds: meanMs,
            medianMilliseconds: medianMs,
            p95Milliseconds: p95Ms,
            top1Index: top1.index,
            top1Label: top1.label,
            top1Probability: top1.probability,
            top5: top5.map { ($0.index, $0.label, $0.probability) },
            notes: notes,
            inputDescription: inputDescription
        )
    }
}

enum BenchmarkSnapshotImporterError: LocalizedError {
    case unsupportedReport

    var errorDescription: String? {
        switch self {
        case .unsupportedReport:
            return "The file does not contain a supported benchmark report."
        }
    }
}

enum BenchmarkSnapshotImporter {
    static func importReport(from data: Data) throws -> BenchmarkReportPayload {
        let decoder = JSONDecoder()
        if let report = try? decoder.decode(BenchmarkReportPayload.self, from: data),
           report.primarySnapshot() != nil {
            return report
        }
        throw BenchmarkSnapshotImporterError.unsupportedReport
    }

    static func importSnapshot(from data: Data) throws -> InferenceSnapshot {
        if let imported = try? importReport(from: data),
           let snapshot = imported.primarySnapshot() {
            return snapshot
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .useDefaultKeys
        return try decoder.decode(BenchmarkSnapshotPayload.self, from: data).toSnapshot()
    }
}
