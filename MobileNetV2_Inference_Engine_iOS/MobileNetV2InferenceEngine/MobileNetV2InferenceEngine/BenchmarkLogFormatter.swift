import Foundation

enum BenchmarkLogFormatter {
    static func format(_ report: BenchmarkReportPayload, as format: ReportFormat) -> String {
        switch format {
        case .markdown:
            return markdown(report)
        case .plainText:
            return plainText(report)
        case .json:
            return json(report)
        }
    }

    static func format(_ snapshot: InferenceSnapshot, as reportFormat: ReportFormat) -> String {
        self.format(ReportDocumentBuilder.makeReport(from: snapshot), as: reportFormat)
    }

    static func plainText(_ report: BenchmarkReportPayload) -> String {
        var lines: [String] = []
        lines.append(report.title)
        lines.append("Environment:")
        for item in report.environment {
            lines.append("- \(item.key): \(item.value)")
        }
        lines.append("Model:")
        for item in report.model {
            lines.append("- \(item.key): \(item.value)")
        }
        lines.append("Latency:")
        for row in report.latencyRows {
            lines.append(String(format: "- %@ / %@ / %@ | load %.3f ms | warmup %d | runs %d | median %.3f ms | p95 %.3f ms | top1 %d %@ | notes %@",
                                row.framework,
                                row.backend,
                                row.device,
                                row.loadMs,
                                row.warmupRuns,
                                row.benchmarkRuns,
                                row.medianMs,
                                row.p95Ms,
                                row.top1Idx,
                                row.top1Label,
                                row.notes))
        }
        lines.append("Result Check:")
        for item in report.resultChecks {
            lines.append("- \(item)")
        }
        lines.append("Engineering Notes:")
        for item in report.engineeringNotes {
            lines.append("- \(item)")
        }
        lines.append("Appendix: Sample Results")
        for (offset, row) in report.sampleRows.enumerated() {
            lines.append(String(format: "- #%d %@ | %@ | expected %@ | top1 %@ | prob %.2f%% | load %.3f ms | mean %.3f ms | median %.3f ms | p95 %.3f ms | %@",
                                offset + 1,
                                row.imageName,
                                row.backend,
                                row.expectedLabel,
                                row.top1Label,
                                row.top1Prob * 100.0,
                                row.loadMs ?? 0.0,
                                row.meanMs,
                                row.medianMs,
                                row.p95Ms,
                                row.note))
        }
        for item in report.appendixNotes {
            lines.append("- \(item)")
        }
        return lines.joined(separator: "\n")
    }

    static func markdown(_ report: BenchmarkReportPayload) -> String {
        var lines: [String] = []
        lines.append("# \(report.title)")
        lines.append("")
        lines.append("## 1. Environment")
        for item in report.environment {
            lines.append("- \(item.key): \(item.value)")
        }
        lines.append("")
        lines.append("## 2. Model")
        for item in report.model {
            lines.append("- \(item.key): \(item.value)")
        }
        lines.append("")
        lines.append("## 3. Latency")
        lines.append("")
        lines.append("| Framework | Backend | Device | Load ms | Warmup | Median ms | P95 ms | Top1 | Notes |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in report.latencyRows {
            lines.append("| \(row.framework) | \(row.backend) | \(row.device) | \(String(format: "%.3f", row.loadMs)) | \(row.warmupRuns) | \(String(format: "%.3f", row.medianMs)) | \(String(format: "%.3f", row.p95Ms)) | \(row.top1Idx) | \(row.notes) |")
        }
        lines.append("")
        lines.append("## 4. Result Check")
        lines.append("")
        for item in report.resultChecks {
            lines.append("- \(item)")
        }
        lines.append("")
        lines.append("## 5. Engineering Notes")
        lines.append("")
        for item in report.engineeringNotes {
            lines.append("- \(item)")
        }
        lines.append("")
        lines.append("## Appendix: Sample Results")
        lines.append("")
        lines.append("| Image | Backend | Expected label | Top-1 prediction | Top-1 prob | Load ms | Mean ms | Median ms | P95 ms | Note |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in report.sampleRows {
            let loadMs = row.loadMs.map { String(format: "%.3f", $0) } ?? "—"
            lines.append("| \(row.imageName) | \(row.backend) | \(row.expectedLabel) | \(row.top1Label) | \(String(format: "%.2f%%", row.top1Prob * 100.0)) | \(loadMs) | \(String(format: "%.3f", row.meanMs)) | \(String(format: "%.3f", row.medianMs)) | \(String(format: "%.3f", row.p95Ms)) | \(row.note) |")
        }
        for item in report.appendixNotes {
            lines.append("- \(item)")
        }
        return lines.joined(separator: "\n")
    }

    static func json(_ report: BenchmarkReportPayload) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        guard let data = try? encoder.encode(report),
              let string = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return string
    }
}
