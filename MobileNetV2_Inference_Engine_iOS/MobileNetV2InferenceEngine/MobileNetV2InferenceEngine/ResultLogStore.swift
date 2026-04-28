import Combine
import Foundation
import UIKit

final class ResultLogStore: ObservableObject {
    @Published var format: ReportFormat = .markdown
    @Published var accumulateLog: Bool = false
    @Published var text: String = ""
    @Published var isPresentingShareSheet: Bool = false
    private var reports: [BenchmarkReportPayload] = []

    func update(with report: BenchmarkReportPayload) {
        if accumulateLog {
            reports.append(report)
        } else {
            reports = [report]
        }
        render()
    }

    func update(with snapshot: InferenceSnapshot) {
        update(with: ReportDocumentBuilder.makeReport(from: snapshot))
    }

    func render() {
        guard !reports.isEmpty else {
            text = ""
            return
        }
        text = reports
            .map { BenchmarkLogFormatter.format($0, as: format) }
            .joined(separator: "\n\n")
    }

    func clear() {
        reports.removeAll()
        text = ""
    }

    func copyToPasteboard() {
        UIPasteboard.general.string = text
    }
}
