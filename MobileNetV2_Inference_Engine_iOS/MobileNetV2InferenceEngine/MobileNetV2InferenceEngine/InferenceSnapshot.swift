import Foundation

struct InferenceSnapshot {
    var framework: String
    var backend: String
    var device: String
    var modelName: String
    var sourceImageName: String
    var debugPreprocessedImagePath: String? = nil
    var loadMilliseconds: Double
    var warmupRuns: Int
    var benchmarkRuns: Int
    var meanMilliseconds: Double
    var medianMilliseconds: Double
    var p95Milliseconds: Double
    var top1Index: Int
    var top1Label: String
    var top1Probability: Double
    var top5: [(index: Int, label: String, probability: Double)]
    var notes: String
    var inputDescription: String
}

enum ReportFormat: String, CaseIterable, Identifiable {
    case markdown = "Markdown"
    case plainText = "Plain Text"
    case json = "JSON"

    var id: String { rawValue }
}
