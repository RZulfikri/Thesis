//
//  ScanHistoryManager.swift
//  TrueDepthStreamer
//
//  Reads, manages and deletes scan folders from the app Documents directory.
//

import Foundation

@MainActor
final class ScanHistoryManager: ObservableObject {

    @Published private(set) var records: [ScanRecord] = []
    @Published private(set) var isLoading = false

    private let fileManager = FileManager.default

    private var documentsURL: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    // MARK: - Load

    func loadRecords() {
        isLoading = true
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            let loaded = await self.scanDisk()
            await MainActor.run {
                self.records = loaded
                self.isLoading = false
            }
        }
    }

    // MARK: - Delete

    func delete(_ record: ScanRecord) {
        try? fileManager.removeItem(at: record.folderURL)
        records.removeAll { $0.id == record.id }
        pruneEmptyLabelFolder(for: record)
    }

    func delete(ids: Set<String>) {
        let toDelete = records.filter { ids.contains($0.id) }
        for record in toDelete {
            try? fileManager.removeItem(at: record.folderURL)
        }
        records.removeAll { ids.contains($0.id) }
        toDelete.forEach { pruneEmptyLabelFolder(for: $0) }
    }

    /// Hapus label folder jika sudah kosong setelah scan di dalamnya dihapus.
    private func pruneEmptyLabelFolder(for record: ScanRecord) {
        let labelFolder = record.folderURL.deletingLastPathComponent()
        if (try? fileManager.contentsOfDirectory(atPath: labelFolder.path))?.isEmpty == true {
            try? fileManager.removeItem(at: labelFolder)
        }
    }

    // MARK: - Private

    private func scanDisk() async -> [ScanRecord] {
        guard let topLevel = try? fileManager.contentsOfDirectory(
            at: documentsURL,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        // Struktur: Documents/[label]/[label_YYYYMMDD_HHMMSS]/
        let scanPattern = #"^.+_\d{8}_\d{6}$"#

        func isScanFolder(_ url: URL) -> Bool {
            (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true &&
            url.lastPathComponent.range(of: scanPattern, options: .regularExpression) != nil
        }

        // Level 1: label folders  →  Level 2: scan folders di dalamnya
        var scanFolders: [URL] = []
        for labelDir in topLevel {
            guard (try? labelDir.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { continue }
            if let subs = try? fileManager.contentsOfDirectory(
                at: labelDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) {
                scanFolders.append(contentsOf: subs.filter { isScanFolder($0) })
            }
        }

        let decoder = JSONDecoder()
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withDashSeparatorInDate,
                             .withColonSeparatorInTime, .withTimeZone]

        var results: [ScanRecord] = []

        for folder in scanFolders {
            let fallbackDate = (try? folder.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? Date.distantPast

            // Fallback label = nama folder parent (Documents/[label]/...)
            var label = folder.deletingLastPathComponent().lastPathComponent
            var frameCount = 0
            var exportDate = fallbackDate

            let metadataURL = folder.appendingPathComponent("metadata.json")
            if let data = try? Data(contentsOf: metadataURL),
               let meta = try? decoder.decode(ScanMetadata.self, from: data) {
                label      = meta.label.isEmpty ? "Unknown" : meta.label
                frameCount = meta.frameCount
                if let parsed = iso.date(from: meta.exportTimestamp) {
                    exportDate = parsed
                }
            }

            results.append(ScanRecord(
                folderURL:  folder,
                folderName: folder.lastPathComponent,
                label:      label,
                frameCount: frameCount,
                exportDate: exportDate
            ))
        }

        // Newest first
        return results.sorted { $0.exportDate > $1.exportDate }
    }
}
