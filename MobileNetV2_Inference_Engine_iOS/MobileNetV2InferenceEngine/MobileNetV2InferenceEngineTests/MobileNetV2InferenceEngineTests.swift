import XCTest
@testable import MobileNetV2InferenceEngine

final class MobileNetV2InferenceEngineTests: XCTestCase {
    private func makeSnapshot(
        framework: String = "MNN",
        backend: String = "CPU",
        device: String = "iPhone 15 Pro",
        top1Index: Int = 258,
        top1Label: String = "Samoyed",
        top1Probability: Double = 0.78
    ) -> InferenceSnapshot {
        InferenceSnapshot(
            framework: framework,
            backend: backend,
            device: device,
            modelName: "MobileNetV2",
            loadMilliseconds: 2.5,
            warmupRuns: 5,
            benchmarkRuns: 20,
            meanMilliseconds: 5.2,
            medianMilliseconds: 5.1,
            p95Milliseconds: 5.4,
            top1Index: top1Index,
            top1Label: top1Label,
            top1Probability: top1Probability,
            top5: [
                (top1Index, top1Label, top1Probability),
            ],
            notes: "measured",
            inputDescription: "1x3x224x224"
        )
    }

    func testNCNNOnlyShowsCPUAndVulkan() throws {
        let config = InferenceConfiguration(framework: .ncnn, backend: .cpu)
        XCTAssertEqual(config.availableBackends, [.cpu, .vulkan])
    }

    func testNCNNMetalNormalizesToVulkan() throws {
        let config = InferenceConfiguration(framework: .ncnn, backend: .metal)
        XCTAssertEqual(config.normalized.backend, .vulkan)
        XCTAssertEqual(config.normalized.displayName, "NCNN / Vulkan")
    }

    func testUnifiedReportExportUsesStableSchema() throws {
        let report = ReportDocumentBuilder.makeReport(from: makeSnapshot())
        let markdown = BenchmarkLogFormatter.markdown(report)
        let json = BenchmarkLogFormatter.json(report)

        XCTAssertTrue(markdown.contains("## 1. Environment"))
        XCTAssertTrue(markdown.contains("## 2. Model"))
        XCTAssertTrue(markdown.contains("## 3. Latency"))
        XCTAssertTrue(markdown.contains("## 4. Result Check"))
        XCTAssertTrue(markdown.contains("## 5. Engineering Notes"))
        XCTAssertTrue(markdown.contains("## Appendix: Sample Results"))
        XCTAssertTrue(markdown.contains("| Image | Backend | Expected label | Top-1 prediction | Top-1 prob | Load ms | Mean ms | Median ms | P95 ms | Note |"))
        XCTAssertTrue(json.contains("\"title\""))
        XCTAssertTrue(json.contains("\"latencyRows\""))
        XCTAssertTrue(json.contains("\"sampleRows\""))

        let decoded = try BenchmarkSnapshotImporter.importReport(from: Data(json.utf8))
        let importedSnapshot = try XCTUnwrap(decoded.primarySnapshot())
        XCTAssertEqual(importedSnapshot.framework, "MNN")
        XCTAssertEqual(importedSnapshot.backend, "CPU")
        XCTAssertEqual(importedSnapshot.top1Label, "Samoyed")
        XCTAssertEqual(importedSnapshot.modelName, "MobileNetV2")
    }

    func testImportMNNBatchReportUsesMeasuredLatencyRow() throws {
        let json = """
        {
          "title": "Week 3 macOS Inference Baseline",
          "environment": [
            { "key": "Device", "value": "Mac M1 Pro" },
            { "key": "Platform", "value": "macOS" }
          ],
          "model": [
            { "key": "Model", "value": "MobileNetV2" },
            { "key": "Input", "value": "1x3x224x224" }
          ],
          "latencyRows": [
            {
              "framework": "MNN",
              "backend": "CPU",
              "device": "Mac M1 Pro",
              "loadMs": 2.5,
              "warmupRuns": 5,
              "benchmarkRuns": 20,
              "meanMs": 5.2,
              "medianMs": 5.1,
              "p95Ms": 5.4,
              "top1Idx": 258,
              "top1Label": "Samoyed",
              "top1Prob": 0.78,
              "notes": "measured"
            },
            {
              "framework": "MNN",
              "backend": "Metal",
              "device": "Mac M1 Pro",
              "loadMs": 0.0,
              "warmupRuns": 0,
              "benchmarkRuns": 0,
              "meanMs": 0.0,
              "medianMs": 0.0,
              "p95Ms": 0.0,
              "top1Idx": -1,
              "top1Label": "",
              "top1Prob": 0.0,
              "notes": "not measured"
            }
          ],
          "resultChecks": [],
          "engineeringNotes": [],
          "sampleRows": [
            {
              "imageName": "test_dog.jpg",
              "backend": "CPU",
              "expectedIdx": 258,
              "expectedLabel": "Samoyed",
              "top1Idx": 258,
              "top1Label": "Samoyed",
              "top1Prob": 0.78,
              "loadMs": 2.5,
              "meanMs": 5.2,
              "medianMs": 5.1,
              "p95Ms": 5.4,
              "note": ""
            }
          ],
          "appendixNotes": [],
          "missingImages": []
        }
        """

        let snapshot = try BenchmarkSnapshotImporter.importSnapshot(from: Data(json.utf8))
        XCTAssertEqual(snapshot.framework, "MNN")
        XCTAssertEqual(snapshot.backend, "CPU")
        XCTAssertEqual(snapshot.device, "Mac M1 Pro")
        XCTAssertEqual(snapshot.modelName, "MobileNetV2")
        XCTAssertEqual(snapshot.top1Index, 258)
        XCTAssertEqual(snapshot.top1Label, "Samoyed")
        XCTAssertEqual(snapshot.warmupRuns, 5)
        XCTAssertEqual(snapshot.benchmarkRuns, 20)
        XCTAssertEqual(snapshot.loadMilliseconds, 2.5, accuracy: 0.0001)
        XCTAssertEqual(snapshot.meanMilliseconds, 5.2, accuracy: 0.0001)
        XCTAssertEqual(snapshot.medianMilliseconds, 5.1, accuracy: 0.0001)
        XCTAssertEqual(snapshot.p95Milliseconds, 5.4, accuracy: 0.0001)
    }

    func testImportNCNNBatchReportUsesMeasuredLatencyRow() throws {
        let json = """
        {
          "title": "Week 3 macOS Inference Baseline",
          "environment": [
            { "key": "Device", "value": "Mac M1 Pro" },
            { "key": "Platform", "value": "macOS" }
          ],
          "model": [
            { "key": "Model", "value": "MobileNetV2" },
            { "key": "Input", "value": "1x3x224x224" }
          ],
          "latencyRows": [
            {
              "framework": "MNN",
              "backend": "CPU",
              "device": "Mac M1 Pro",
              "loadMs": 0.0,
              "warmupRuns": 0,
              "benchmarkRuns": 0,
              "meanMs": 0.0,
              "medianMs": 0.0,
              "p95Ms": 0.0,
              "top1Idx": -1,
              "top1Label": "",
              "top1Prob": 0.0,
              "notes": "not measured"
            },
            {
              "framework": "NCNN",
              "backend": "CPU",
              "device": "Mac M1 Pro",
              "loadMs": 3.97,
              "warmupRuns": 5,
              "benchmarkRuns": 20,
              "meanMs": 5.11,
              "medianMs": 5.12,
              "p95Ms": 5.39,
              "top1Idx": 258,
              "top1Label": "Samoyed",
              "top1Prob": 0.654,
              "notes": "measured"
            }
          ],
          "resultChecks": [],
          "engineeringNotes": [],
          "sampleRows": [
            {
              "imageName": "test_dog.jpg",
              "backend": "CPU",
              "expectedIdx": 258,
              "expectedLabel": "Samoyed",
              "top1Idx": 258,
              "top1Label": "Samoyed",
              "top1Prob": 0.654,
              "loadMs": 3.97,
              "meanMs": 5.11,
              "medianMs": 5.12,
              "p95Ms": 5.39,
              "note": ""
            }
          ],
          "appendixNotes": [],
          "missingImages": []
        }
        """

        let snapshot = try BenchmarkSnapshotImporter.importSnapshot(from: Data(json.utf8))
        XCTAssertEqual(snapshot.framework, "NCNN")
        XCTAssertEqual(snapshot.backend, "CPU")
        XCTAssertEqual(snapshot.device, "Mac M1 Pro")
        XCTAssertEqual(snapshot.modelName, "MobileNetV2")
        XCTAssertEqual(snapshot.top1Index, 258)
        XCTAssertEqual(snapshot.top1Label, "Samoyed")
        XCTAssertEqual(snapshot.warmupRuns, 5)
        XCTAssertEqual(snapshot.benchmarkRuns, 20)
        XCTAssertEqual(snapshot.loadMilliseconds, 3.97, accuracy: 0.0001)
        XCTAssertEqual(snapshot.meanMilliseconds, 5.11, accuracy: 0.0001)
        XCTAssertEqual(snapshot.medianMilliseconds, 5.12, accuracy: 0.0001)
        XCTAssertEqual(snapshot.p95Milliseconds, 5.39, accuracy: 0.0001)
    }
}
