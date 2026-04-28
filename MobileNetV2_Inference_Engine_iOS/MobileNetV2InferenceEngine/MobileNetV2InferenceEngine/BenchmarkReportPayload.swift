import Foundation

struct BenchmarkReportPayload: Codable {
    struct KeyValue: Codable {
        let key: String
        let value: String
    }

    struct LatencyRow: Codable {
        let framework: String
        let backend: String
        let device: String
        let loadMs: Double
        let warmupRuns: Int
        let benchmarkRuns: Int
        let meanMs: Double
        let medianMs: Double
        let p95Ms: Double
        let top1Idx: Int
        let top1Label: String
        let top1Prob: Double
        let notes: String
    }

    struct SampleRow: Codable {
        let imageName: String
        let backend: String
        let expectedIdx: Int
        let expectedLabel: String
        let top1Idx: Int
        let top1Label: String
        let top1Prob: Double
        let loadMs: Double?
        let meanMs: Double
        let medianMs: Double
        let p95Ms: Double
        let note: String
    }

    let title: String
    let environment: [KeyValue]
    let model: [KeyValue]
    let latencyRows: [LatencyRow]
    let resultChecks: [String]
    let engineeringNotes: [String]
    let sampleRows: [SampleRow]
    let appendixNotes: [String]
}
