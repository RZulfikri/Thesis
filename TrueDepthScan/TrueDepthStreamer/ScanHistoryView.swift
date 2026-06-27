//
//  ScanHistoryView.swift
//  TrueDepthStreamer
//
//  Lists all saved scans with per-row share/delete and multi-select support.
//

import SwiftUI
import UIKit

struct ScanHistoryView: View {

    @StateObject private var manager = ScanHistoryManager()
    @ObservedObject private var queue = ProcessingQueue.shared
    @Environment(\.dismiss) private var dismiss

    @State private var isSelecting = false
    @State private var selectedIDs: Set<String> = []


    // Single-row delete confirmation
    @State private var recordPendingDelete: ScanRecord?
    @State private var showSingleDeleteAlert = false

    // Multi-select delete confirmation
    @State private var showBatchDeleteAlert = false

    // MARK: - Body

    var body: some View {
        ZStack {
        NavigationView {
            Group {
                if manager.isLoading {
                    ProgressView("Loading…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if manager.records.isEmpty {
                    VStack(spacing: 14) {
                        Image(systemName: "clock.arrow.circlepath")
                            .font(.system(size: 52))
                            .foregroundColor(.secondary)
                        Text("No scans yet")
                            .font(.title3)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(selection: isSelecting ? $selectedIDs : .constant(Set<String>())) {
                        // Bagian "Sedang Diproses" — task yang belum selesai di background
                        let activeTasks = queue.tasks.filter { $0.status.isActive }
                        if !activeTasks.isEmpty {
                            Section {
                                ForEach(activeTasks) { task in
                                    ProcessingTaskRowView(task: task)
                                }
                            } header: {
                                HStack(spacing: 6) {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle())
                                        .scaleEffect(0.7)
                                    Text("Sedang Diproses")
                                        .font(.caption.weight(.semibold))
                                }
                            }
                        }

                        // Scan tersimpan — dikelompokkan per label (A–Z)
                        let grouped = Dictionary(grouping: manager.records, by: \.label)
                        ForEach(grouped.keys.sorted(), id: \.self) { label in
                            let groupRecords = grouped[label]!
                            Section {
                                ForEach(groupRecords) { record in
                                    ScanRowView(
                                        record: record,
                                        isSelecting: isSelecting,
                                        onShare: { shareRecords([record]) },
                                        onDelete: {
                                            recordPendingDelete = record
                                            showSingleDeleteAlert = true
                                        }
                                    )
                                }
                            } header: {
                                GroupHeaderView(
                                    label: label,
                                    count: groupRecords.count,
                                    isSelecting: isSelecting,
                                    selectionState: groupSelectionState(groupRecords),
                                    onToggle: { toggleGroupSelection(groupRecords) }
                                )
                            }
                        }
                    }
                    .environment(\.editMode, isSelecting ? .constant(.active) : .constant(.inactive))
                }
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbarContent }
            // Bottom bar for batch actions
            .safeAreaInset(edge: .bottom) {
                if isSelecting && !manager.records.isEmpty {
                    batchActionBar
                }
            }
        }
        .onAppear { manager.loadRecords() }
        // Full-screen zipping overlay
        if isZipping {
            Color.black.opacity(0.5)
                .ignoresSafeArea()
            VStack(spacing: 16) {
                ProgressView()
                    .progressViewStyle(.circular)
                    .scaleEffect(1.4)
                    .tint(.white)
                Text("Preparing files…")
                    .foregroundColor(.white)
                    .font(.subheadline)
            }
        }
        } // ZStack
        // Single delete confirmation
        .alert("Delete Scan", isPresented: $showSingleDeleteAlert, presenting: recordPendingDelete) { record in
            Button("Delete", role: .destructive) {
                manager.delete(record)
            }
            Button("Cancel", role: .cancel) {}
        } message: { record in
            Text("\"\(record.label)\" will be permanently deleted.")
        }
        // Batch delete confirmation
        .alert("Delete \(selectedIDs.count) Scan\(selectedIDs.count == 1 ? "" : "s")",
               isPresented: $showBatchDeleteAlert) {
            Button("Delete", role: .destructive) {
                manager.delete(ids: selectedIDs)
                selectedIDs.removeAll()
                if manager.records.isEmpty { isSelecting = false }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The selected scans will be permanently deleted.")
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigationBarLeading) {
            Button(isSelecting ? "Cancel" : "Close") {
                if isSelecting {
                    isSelecting = false
                    selectedIDs.removeAll()
                } else {
                    dismiss()
                }
            }
        }
        ToolbarItem(placement: .navigationBarTrailing) {
            if !manager.records.isEmpty {
                Button(isSelecting ? "Done" : "Select") {
                    isSelecting.toggle()
                    if !isSelecting { selectedIDs.removeAll() }
                }
            }
        }
    }

    // MARK: - Batch action bar

    private var batchActionBar: some View {
        HStack(spacing: 24) {
            // Delete selected
            Button {
                showBatchDeleteAlert = true
            } label: {
                Label("Delete (\(selectedIDs.count))", systemImage: "trash")
                    .font(.subheadline.weight(.medium))
            }
            .disabled(selectedIDs.isEmpty)
            .tint(.red)

            Spacer()

            // Share selected
            Button {
                let toShare = manager.records.filter { selectedIDs.contains($0.id) }
                shareRecords(toShare)
            } label: {
                Label("Share (\(selectedIDs.count))", systemImage: "square.and.arrow.up")
                    .font(.subheadline.weight(.medium))
            }
            .disabled(selectedIDs.isEmpty || isZipping)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
    }

    // MARK: - Group Selection Helpers

    enum GroupSelectionState { case none, partial, all }

    private func groupSelectionState(_ records: [ScanRecord]) -> GroupSelectionState {
        let n = records.filter { selectedIDs.contains($0.id) }.count
        if n == 0              { return .none }
        if n == records.count  { return .all }
        return .partial
    }

    private func toggleGroupSelection(_ records: [ScanRecord]) {
        if groupSelectionState(records) == .all {
            records.forEach { selectedIDs.remove($0.id) }
        } else {
            records.forEach { selectedIDs.insert($0.id) }
        }
    }

    // MARK: - Share

    @State private var isZipping = false

    private func shareRecords(_ records: [ScanRecord]) {
        guard !isZipping else { return }
        isZipping = true
        let folderURLs = records.map { $0.folderURL }
        let zipName: String = {
            if records.count == 1 {
                return records[0].folderURL.lastPathComponent
            }
            let stamp = DateFormatter.localizedString(from: Date(), dateStyle: .short, timeStyle: .short)
                .replacingOccurrences(of: "/", with: "-")
                .replacingOccurrences(of: ":", with: "-")
                .replacingOccurrences(of: " ", with: "_")
            return "scans_\(stamp)"
        }()
        DispatchQueue.global(qos: .userInitiated).async {
            let zipURL = ContentView.zipFolders(folderURLs, zipName: zipName)
            DispatchQueue.main.async {
                self.isZipping = false
                guard let zipURL else { return }
                let vc = UIActivityViewController(activityItems: [zipURL], applicationActivities: nil)
                vc.completionWithItemsHandler = { _, _, _, _ in
                    try? FileManager.default.removeItem(at: zipURL)
                }
                guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                      let root = windowScene.windows.first(where: { $0.isKeyWindow })?.rootViewController else { return }
                var top = root
                while let presented = top.presentedViewController { top = presented }
                top.present(vc, animated: true)
            }
        }
    }
}

// MARK: - Group Header View

private struct GroupHeaderView: View {
    let label: String
    let count: Int
    let isSelecting: Bool
    let selectionState: ScanHistoryView.GroupSelectionState
    let onToggle: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            if isSelecting {
                Button(action: onToggle) {
                    Image(systemName: checkboxIcon)
                        .font(.system(size: 18))
                        .foregroundColor(selectionState == .none ? .secondary : .accentColor)
                        .animation(.easeInOut(duration: 0.15), value: selectionState)
                }
                .buttonStyle(.plain)
            }

            Text(label)
                .font(.subheadline.weight(.semibold))
            Text("(\(count))")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()
        }
        .contentShape(Rectangle())      // seluruh area header bisa di-tap
        .onTapGesture { if isSelecting { onToggle() } }
    }

    private var checkboxIcon: String {
        switch selectionState {
        case .none:    return "circle"
        case .partial: return "minus.circle.fill"
        case .all:     return "checkmark.circle.fill"
        }
    }
}

// MARK: - Processing Task Row (scan yang sedang diproses, belum tersimpan)

private struct ProcessingTaskRowView: View {
    let task: ProcessingTask

    var body: some View {
        HStack(spacing: 12) {
            // Spinner kecil
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle())
                .scaleEffect(0.85)
                .frame(width: 24, height: 24)

            VStack(alignment: .leading, spacing: 3) {
                Text(task.label)
                    .font(.headline)
                Text(task.status.displayText)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text(task.timestamp)
                .font(.caption.monospacedDigit())
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Row View

private struct ScanRowView: View {
    let record: ScanRecord
    let isSelecting: Bool
    let onShare: () -> Void
    let onDelete: () -> Void

    /// "rahmat_20260505_134500" → "20260505_134500"
    private var sessionID: String {
        let prefix = record.label + "_"
        guard record.folderName.hasPrefix(prefix) else { return record.folderName }
        return String(record.folderName.dropFirst(prefix.count))
    }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(sessionID)
                    .font(.subheadline.monospacedDigit())
                Text("\(record.frameCount) frames")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Action buttons — hidden while in select mode
            if !isSelecting {
                HStack(spacing: 4) {
                    Button(action: onShare) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 17))
                            .frame(width: 36, height: 36)
                    }
                    .tint(.blue)
                    .buttonStyle(.borderless)

                    Button(action: onDelete) {
                        Image(systemName: "trash")
                            .font(.system(size: 17))
                            .frame(width: 36, height: 36)
                    }
                    .tint(.red)
                    .buttonStyle(.borderless)
                }
            }
        }
        .padding(.vertical, 4)
    }
}
