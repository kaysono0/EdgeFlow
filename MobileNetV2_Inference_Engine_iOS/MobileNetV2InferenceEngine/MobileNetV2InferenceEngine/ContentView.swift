import SwiftUI
import PhotosUI
import Photos
import UniformTypeIdentifiers
import UIKit

struct ContentView: View {
    @StateObject private var viewModel = InferenceViewModel()
    @StateObject private var logStore = ResultLogStore()
    @State private var isImporting = false
    @State private var isPresentingTestImageSourceDialog = false
    @State private var isImportingTestImage = false
    @State private var isPickingTestImageFromPhotos = false
    @State private var selectedPhotoItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    headerCard
                    inputCard
                    outputCard
                    sourceCard
                    logCard
                }
                .padding()
            }
            .navigationTitle("MobileNetV2 Inference")
            .onAppear {
                logStore.update(with: viewModel.report)
            }
            .onChange(of: viewModel.runToken) { _, _ in
                logStore.update(with: viewModel.report)
            }
            .onChange(of: logStore.format) { _, _ in
                logStore.render()
            }
            .confirmationDialog(
                "Select Test Image Source",
                isPresented: $isPresentingTestImageSourceDialog,
                titleVisibility: .visible
            ) {
                Button("Photos") {
                    isPickingTestImageFromPhotos = true
                }
                Button("Files") {
                    isImportingTestImage = true
                }
                Button("Cancel", role: .cancel) {}
            }
        }
        .sheet(isPresented: $logStore.isPresentingShareSheet) {
            ShareSheet(items: [logStore.text])
        }
        .photosPicker(
            isPresented: $isPickingTestImageFromPhotos,
            selection: $selectedPhotoItem,
            matching: .images,
            preferredItemEncoding: .current
        )
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [UTType.json, UTType.text, UTType.plainText],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                guard let data = try? Data(contentsOf: url) else { return }
                viewModel.importSnapshot(from: data, sourceName: url.lastPathComponent)
                logStore.update(with: viewModel.report)
            case .failure:
                break
            }
        }
        .fileImporter(
            isPresented: $isImportingTestImage,
            allowedContentTypes: [UTType.image],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                do {
                    let copiedURL = try copyImageToTemporaryLocation(from: url)
                    viewModel.selectTestImage(at: copiedURL.path, sourceName: url.lastPathComponent)
                } catch {
                    viewModel.statusText = "Failed to import test image: \(url.lastPathComponent)"
                }
            case .failure:
                break
            }
        }
        .onChange(of: selectedPhotoItem) { _, newValue in
            guard let item = newValue else { return }
            Task {
                do {
                    guard let data = try await item.loadTransferable(type: Data.self) else {
                        await MainActor.run {
                            viewModel.statusText = "Failed to load selected photo"
                        }
                        return
                    }
                    let copiedURL = try writeImageDataToTemporaryLocation(data)
                    await MainActor.run {
                        viewModel.selectTestImage(at: copiedURL.path, sourceName: copiedURL.lastPathComponent)
                    }
                } catch {
                    await MainActor.run {
                        viewModel.statusText = "Failed to import photo from Photos"
                    }
                }
            }
        }
    }

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Inference Runner")
                .font(.title2.bold())
            Text(viewModel.statusText)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 10) {
                Picker("Framework", selection: Binding(
                    get: { viewModel.selectedConfiguration.framework },
                    set: { viewModel.selectFramework($0) }
                )) {
                    ForEach(InferenceFramework.allCases) { framework in
                        Text(framework.rawValue).tag(framework)
                    }
                }
                .pickerStyle(.segmented)

                Picker("Backend", selection: Binding(
                    get: { viewModel.selectedConfiguration.backend },
                    set: { viewModel.selectBackend($0) }
                )) {
                    ForEach(viewModel.selectedConfiguration.availableBackends) { backend in
                        Text(backend.rawValue).tag(backend)
                    }
                }
                .pickerStyle(.segmented)
            }
            Text(viewModel.selectedConfiguration.runnerNotes)
                .font(.footnote)
                .foregroundStyle(.secondary)
            Picker("Export Format", selection: $logStore.format) {
                ForEach(ReportFormat.allCases) { format in
                    Text(format.rawValue).tag(format)
                }
            }
            .pickerStyle(.segmented)

            Divider()
                .padding(.vertical, 4)

            actionRow
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }

    private var inputCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Input")
                .font(.headline)
            Text("Current test image: \(viewModel.selectedTestImageName)")
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
            previewImageCard
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }

    private var outputCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Output")
                .font(.headline)

            VStack(alignment: .leading, spacing: 10) {
                metricRow("Framework", viewModel.snapshot.framework)
                metricRow("Backend", viewModel.snapshot.backend)
                metricRow("Device", viewModel.snapshot.device)
                metricRow("Top1", "\(viewModel.snapshot.top1Index) \(viewModel.snapshot.top1Label)")
                metricRow("Latency", String(format: "mean %.3f ms, median %.3f ms, p95 %.3f ms",
                                             viewModel.snapshot.meanMilliseconds,
                                             viewModel.snapshot.medianMilliseconds,
                                             viewModel.snapshot.p95Milliseconds))

                VStack(alignment: .leading, spacing: 6) {
                    Text("Top5")
                        .font(.subheadline.bold())
                    ForEach(Array(viewModel.snapshot.top5.enumerated()), id: \.offset) { element in
                        let index = element.offset
                        let item = element.element
                        HStack {
                            Text("#\(index + 1)")
                                .frame(width: 32, alignment: .leading)
                                .foregroundStyle(.secondary)
                            Text("\(item.index)")
                                .frame(width: 48, alignment: .leading)
                            Text(item.label)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Text(String(format: "%.2f%%", item.probability * 100.0))
                                .monospacedDigit()
                        }
                        .font(.footnote)
                    }
                }
                .padding(.top, 4)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }

    private var sourceCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Source")
                .font(.headline)
            Text(viewModel.loadedSource)
                .font(.subheadline.monospaced())
            Text("Test image: \(viewModel.selectedTestImageName)")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text(viewModel.statusText)
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text(viewModel.selectedConfiguration.runnerNotes)
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text("Current config: \(viewModel.selectedConfiguration.displayName)")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }

    private var logCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Log")
                .font(.headline)
            Text(logStore.text)
                .font(.system(.footnote, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
                .padding()
                .background(.black.opacity(0.06), in: RoundedRectangle(cornerRadius: 16))
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }

    private var actionRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Button {
                    viewModel.run()
                } label: {
                    Label(viewModel.isRunning ? "Running" : "Run", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isRunning)

                Button {
                    logStore.copyToPasteboard()
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                .buttonStyle(.bordered)

                Button {
                    logStore.isPresentingShareSheet = true
                } label: {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.bordered)
            }

            HStack(spacing: 12) {
                Button {
                    isImporting = true
                } label: {
                    Label("Import", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)

                Button {
                    isPresentingTestImageSourceDialog = true
                } label: {
                    Label("Test Image", systemImage: "photo.on.rectangle.angled")
                }
                .buttonStyle(.bordered)
            }

            Toggle("Accumulate Log", isOn: Binding(
                get: { logStore.accumulateLog },
                set: { newValue in
                    logStore.accumulateLog = newValue
                    logStore.clear()
                }
            ))
            .toggleStyle(.switch)
        }
    }

    private var previewImageCard: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18)
                .fill(
                    LinearGradient(
                        colors: [.indigo.opacity(0.22), .cyan.opacity(0.18)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
            if let uiImage = currentPreviewUIImage {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: 240)
                    .clipShape(RoundedRectangle(cornerRadius: 18))
                    .padding(8)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "photo")
                        .font(.system(size: 46, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text("No preview available")
                        .font(.headline)
                    Text("Tap Test Image to import a photo.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 32)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 240)
        .clipShape(RoundedRectangle(cornerRadius: 18))
    }

    private func metricRow(_ key: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(key)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)
            Text(value)
                .font(.subheadline.monospaced())
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var currentPreviewUIImage: UIImage? {
        guard let path = viewModel.selectedTestImagePath else {
            return nil
        }
        return UIImage(contentsOfFile: path)
    }

    private func writeImageDataToTemporaryLocation(_ data: Data) throws -> URL {
        let tempDirectory = FileManager.default.temporaryDirectory
        let fileURL = tempDirectory.appendingPathComponent("test-image-\(UUID().uuidString).jpg")
        try data.write(to: fileURL, options: [.atomic])
        return fileURL
    }

    private func copyImageToTemporaryLocation(from sourceURL: URL) throws -> URL {
        let tempDirectory = FileManager.default.temporaryDirectory
        let destinationURL = tempDirectory.appendingPathComponent("test-image-\(UUID().uuidString)-\(sourceURL.lastPathComponent)")
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        try fileManager.copyItem(at: sourceURL, to: destinationURL)
        return destinationURL
    }
}
