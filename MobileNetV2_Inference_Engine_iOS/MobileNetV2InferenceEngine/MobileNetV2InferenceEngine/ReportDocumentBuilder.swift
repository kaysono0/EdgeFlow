import Foundation
import UIKit

enum ReportDocumentBuilder {
    static func makeReport(from snapshot: InferenceSnapshot) -> BenchmarkReportPayload {
        let framework = snapshot.framework
        let backend = snapshot.backend
        let deviceName = UIDevice.current.name
        let osVersion = "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)"
        let top1Backend = "\(framework) \(backend)"

        return BenchmarkReportPayload(
            title: "Week 03 iOS Inference Baseline",
            environment: [
                .init(key: "设备", value: deviceName),
                .init(key: "Platform", value: "iOS"),
                .init(key: "OS", value: osVersion),
                .init(key: "Runtime", value: framework),
                .init(key: "Backend", value: backend),
                .init(key: "测试轮数", value: "\(snapshot.benchmarkRuns)"),
                .init(key: "预热轮数", value: "\(snapshot.warmupRuns)"),
            ],
            model: [
                .init(key: "Model", value: snapshot.modelName),
                .init(key: "Test Image", value: snapshot.sourceImageName),
                .init(key: "Source", value: "Native runner output"),
                .init(key: "Input", value: snapshot.inputDescription),
                .init(key: "Output", value: "1x1000"),
                .init(key: "Preprocess", value: "ImageNet resize / crop / normalize"),
            ],
            latencyRows: [
                .init(
                    framework: framework,
                    backend: backend,
                    device: deviceName,
                    loadMs: snapshot.loadMilliseconds,
                    warmupRuns: snapshot.warmupRuns,
                    benchmarkRuns: snapshot.benchmarkRuns,
                    meanMs: snapshot.meanMilliseconds,
                    medianMs: snapshot.medianMilliseconds,
                    p95Ms: snapshot.p95Milliseconds,
                    top1Idx: snapshot.top1Index,
                    top1Label: snapshot.top1Label,
                    top1Prob: snapshot.top1Probability,
                    notes: snapshot.notes
                )
            ],
            resultChecks: [
                "Top-1 label: \(snapshot.top1Label)",
                String(format: "Top-1 prob: %.2f%%", snapshot.top1Probability * 100.0),
            ],
            engineeringNotes: [
                "This report is snapshot-derived on iOS.",
                "Copy and share use the same week03 report schema as macOS.",
            ],
            sampleRows: [
                .init(
                    imageName: snapshot.sourceImageName,
                    backend: top1Backend,
                    expectedIdx: snapshot.top1Index,
                    expectedLabel: snapshot.top1Label,
                    top1Idx: snapshot.top1Index,
                    top1Label: snapshot.top1Label,
                    top1Prob: snapshot.top1Probability,
                    loadMs: snapshot.loadMilliseconds,
                    meanMs: snapshot.meanMilliseconds,
                    medianMs: snapshot.medianMilliseconds,
                    p95Ms: snapshot.p95Milliseconds,
                    note: snapshot.notes
                )
            ],
            appendixNotes: [
                "Use Copy or Share to move this report back to macOS.",
                "The JSON keys match the week03 baseline schema.",
            ]
        )
    }
}

extension BenchmarkReportPayload {
    func primaryLatencyRow() -> LatencyRow? {
        if let measured = latencyRows.first(where: { $0.benchmarkRuns > 0 || $0.meanMs > 0 || $0.top1Idx >= 0 }) {
            return measured
        }
        return latencyRows.first
    }

    func primarySampleRow() -> SampleRow? {
        if let measured = sampleRows.first(where: { $0.top1Idx >= 0 || $0.meanMs > 0 || $0.medianMs > 0 }) {
            return measured
        }
        return sampleRows.first
    }

    func primarySnapshot() -> InferenceSnapshot? {
        let latency = primaryLatencyRow()
        let sample = primarySampleRow()

        if let sample {
            let top5 = [
                (sample.top1Idx, sample.top1Label, sample.top1Prob),
            ]
            return InferenceSnapshot(
                framework: latency?.framework ?? resolvedFramework(for: sample.backend),
                backend: latency?.backend ?? resolvedBackend(for: sample.backend),
                device: latency?.device ?? environmentValue(for: ["设备", "Device"]) ?? "iPhone",
                modelName: model.first(where: { $0.key == "Model" })?.value ?? "MobileNetV2",
                sourceImageName: sample.imageName,
                loadMilliseconds: sample.loadMs ?? latency?.loadMs ?? 0.0,
                warmupRuns: latency?.warmupRuns ?? 0,
                benchmarkRuns: latency?.benchmarkRuns ?? 0,
                meanMilliseconds: sample.meanMs,
                medianMilliseconds: sample.medianMs,
                p95Milliseconds: sample.p95Ms,
                top1Index: sample.top1Idx,
                top1Label: sample.top1Label,
                top1Probability: sample.top1Prob,
                top5: top5,
                notes: sample.note.isEmpty ? (latency?.notes ?? "") : sample.note,
                inputDescription: model.first(where: { $0.key == "Input" })?.value ?? "1x3x224x224"
            )
        }

        if let latency {
            return InferenceSnapshot(
                framework: latency.framework,
                backend: latency.backend,
                device: latency.device,
                modelName: model.first(where: { $0.key == "Model" })?.value ?? "MobileNetV2",
                sourceImageName: sample?.imageName ?? "samoyed.jpg",
                loadMilliseconds: latency.loadMs,
                warmupRuns: latency.warmupRuns,
                benchmarkRuns: latency.benchmarkRuns,
                meanMilliseconds: latency.meanMs,
                medianMilliseconds: latency.medianMs,
                p95Milliseconds: latency.p95Ms,
                top1Index: latency.top1Idx,
                top1Label: latency.top1Label,
                top1Probability: latency.top1Prob,
                top5: [
                    (latency.top1Idx, latency.top1Label, latency.top1Prob),
                ],
                notes: latency.notes,
                inputDescription: model.first(where: { $0.key == "Input" })?.value ?? "1x3x224x224"
            )
        }

        return nil
    }

    private func environmentValue(for keys: [String]) -> String? {
        for key in keys {
            if let value = environment.first(where: { $0.key == key })?.value, !value.isEmpty {
                return value
            }
        }
        return nil
    }

    private func resolvedFramework(for backendValue: String) -> String {
        if backendValue.contains("NCNN") {
            return "NCNN"
        }
        if backendValue.contains("MNN") {
            return "MNN"
        }
        return environment.first(where: { $0.key == "Runtime" })?.value ?? "MNN"
    }

    private func resolvedBackend(for backendValue: String) -> String {
        if backendValue.contains("Metal") {
            return "Metal"
        }
        if backendValue.contains("CPU") {
            return "CPU"
        }
        return environment.first(where: { $0.key == "Backend" })?.value ?? "CPU"
    }
}
