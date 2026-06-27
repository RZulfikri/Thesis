//
//  ScanRecord.swift
//  TrueDepthStreamer
//
//  Lightweight value type representing a single saved scan folder on disk.
//

import Foundation

struct ScanRecord: Identifiable {
    /// Stable identity across list refreshes — the folder's absolute path.
    var id: String { folderURL.path }

    let folderURL: URL       // e.g. .../Documents/rahmat_20260401_134500/
    let folderName: String   // "rahmat_20260401_134500"
    let label: String        // from metadata.json; "Unknown" for old scans without label
    let frameCount: Int      // total depth frames in the scan
    let exportDate: Date     // parsed from metadata.exportTimestamp; folder mtime as fallback
}
