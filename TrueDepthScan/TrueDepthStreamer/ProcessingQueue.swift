//
//  ProcessingQueue.swift
//  TrueDepthStreamer
//
//  Mengelola antrian background processing untuk setiap scan yang selesai.
//  Non-blocking: UI tetap responsif selama export berjalan di worker queue.
//  User bisa scan lagi segera setelah capture selesai.
//

import Foundation
import CoreVideo
import AVFoundation
import simd
import os.log

private let pqLog = OSLog(
    subsystem: Bundle.main.bundleIdentifier ?? "TrueDepthStreamer",
    category: "ProcessingQueue"
)

// MARK: - Captured Scan Data Bundle

/// Semua data yang dibutuhkan untuk post-scan processing.
/// Di-bundle setelah snapshot dari dataOutputQueue, lalu dikirim ke ProcessingQueue.
struct CapturedScanData {
    let label: String
    let timestamp: String
    let folderName: String              // "\(label)_\(timestamp)"
    let points: [simd_float3]           // Voxelized point cloud
    let colors: [simd_uchar3]           // Colors per point
    let depthFrames: [CVPixelBuffer]    // Raw depth frames (retained)
    let depthROIs: [CGRect]             // Bounding-box ROI per frame — Vision normalized coords
    let depthLandmarks: [[CGPoint]]     // Hand joint landmarks per frame — Vision normalized coords
    let calibrationData: AVCameraCalibrationData?
    let handedness: String              // "right", "left", "unknown"
}

// MARK: - Processing Status

enum ProcessingStatus: Equatable {
    case queued                     // Menunggu giliran di queue
    case processing(String)         // Sedang berjalan — teks fase aktif
    case done                       // Selesai sukses
    case failed(String)             // Gagal — pesan error

    static func == (lhs: ProcessingStatus, rhs: ProcessingStatus) -> Bool {
        switch (lhs, rhs) {
        case (.queued, .queued):                               return true
        case (.done, .done):                                   return true
        case (.processing(let a), .processing(let b)):         return a == b
        case (.failed(let a), .failed(let b)):                 return a == b
        default:                                               return false
        }
    }

    var isActive: Bool {
        switch self {
        case .queued, .processing: return true
        default:                   return false
        }
    }

    var displayText: String {
        switch self {
        case .queued:                return "Menunggu..."
        case .processing(let phase): return phase
        case .done:                  return "Selesai"
        case .failed(let msg):       return "Gagal: \(msg)"
        }
    }
}

// MARK: - Processing Task

struct ProcessingTask: Identifiable {
    let id: UUID
    let label: String       // "rahmat"
    let timestamp: String   // "20260401_134500"
    let folderName: String  // "rahmat_20260401_134500"
    var status: ProcessingStatus
    let createdAt: Date

    init(label: String, timestamp: String, folderName: String) {
        self.id         = UUID()
        self.label      = label
        self.timestamp  = timestamp
        self.folderName = folderName
        self.status     = .queued
        self.createdAt  = Date()
    }
}

// MARK: - Processing Queue

/// Singleton ObservableObject yang mengelola antrian export scan.
///
/// Semua mutasi `tasks` terjadi di main thread.
/// Kerja berat (PLY + Open3D export) berjalan di `workerQueue` (utility QoS, concurrent).
/// Jumlah task yang berjalan bersamaan dibatasi oleh `maxConcurrentTasks`.
class ProcessingQueue: ObservableObject {

    static let shared = ProcessingQueue()
    private init() {}

    /// Antrian task — diobservasi oleh UI.
    @Published private(set) var tasks: [ProcessingTask] = []

    /// Maksimum task yang berjalan bersamaan.
    /// 2 = seimbang antara throughput dan beban CPU/I/O iPhone.
    var maxConcurrentTasks: Int {
        get { workerQueue.maxConcurrentOperationCount }
        set { workerQueue.maxConcurrentOperationCount = max(1, newValue) }
    }

    /// Concurrent operation queue — beberapa export bisa berjalan bersamaan.
    /// QoS .utility agar tidak berkompetisi dengan AVCaptureSession (.userInitiated).
    private let workerQueue: OperationQueue = {
        let q = OperationQueue()
        q.name = "com.thesis.processingqueue.worker"
        q.qualityOfService = .utility
        q.maxConcurrentOperationCount = 2   // default: maks 2 task sekaligus
        return q
    }()

    // MARK: - Public API

    var hasActiveTasks: Bool {
        tasks.contains { $0.status.isActive }
    }

    var activeCount: Int {
        tasks.filter { $0.status.isActive }.count
    }

    /// Tambahkan scan ke antrian. Harus dipanggil dari main thread.
    func enqueue(_ data: CapturedScanData) {
        let task = ProcessingTask(
            label: data.label,
            timestamp: data.timestamp,
            folderName: data.folderName
        )
        tasks.append(task)
        let taskID = task.id
        os_log(.info, log: pqLog, "📥 Enqueued: %{public}@", data.folderName)

        workerQueue.addOperation { [weak self] in
            self?.execute(taskID: taskID, data: data)
        }
    }

    /// Hapus task yang sudah selesai/gagal secara manual.
    func clearCompleted() {
        tasks.removeAll { !$0.status.isActive }
    }

    /// Hapus satu task failed — dipanggil saat user tap tombol dismiss.
    func dismissFailed(_ taskID: UUID) {
        guard let idx = tasks.firstIndex(where: { $0.id == taskID }),
              case .failed = tasks[idx].status else { return }
        tasks.remove(at: idx)
    }

    // MARK: - Private Worker (runs on workerQueue)

    private func setStatus(_ status: ProcessingStatus, for taskID: UUID) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            if let idx = self.tasks.firstIndex(where: { $0.id == taskID }) {
                self.tasks[idx].status = status
            }
        }
    }

    private func removeTask(_ taskID: UUID) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            if let idx = self.tasks.firstIndex(where: { $0.id == taskID }) {
                self.tasks.remove(at: idx)
            }
        }
    }

    private func execute(taskID: UUID, data: CapturedScanData) {
        os_log(.info, log: pqLog, "▶️ Start processing: %{public}@", data.folderName)

        // Phase 1: PLY export (opsional — untuk debug/visualisasi point cloud)
        setStatus(.processing("Menyimpan PLY..."), for: taskID)
        _ = D3ProcessingService.saveAccumulatedPLY(points: data.points, colors: data.colors)

        // Phase 2: Open3D depth frame export (file utama untuk pipeline Python)
        setStatus(.processing("Mengekspor frame depth..."), for: taskID)
        let exportURL = D3ProcessingService.exportForOpen3D(
            depthFrames:        data.depthFrames,
            roiPerFrame:        data.depthROIs,
            landmarksPerFrame:  data.depthLandmarks,
            calibrationData:    data.calibrationData,
            folderName:         data.folderName,
            label:              data.label,
            handedness:         data.handedness
        )

        if exportURL != nil {
            os_log(.info, log: pqLog, "✅ Done: %{public}@", data.folderName)
            setStatus(.done, for: taskID)
            // Auto-hapus dari list setelah 5 detik (dengan animasi)
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
                self?.removeTask(taskID)
            }
        } else {
            os_log(.error, log: pqLog, "❌ Failed: %{public}@", data.folderName)
            setStatus(.failed("Export gagal"), for: taskID)
            // Task gagal tetap tampil sampai user dismiss manual
        }
    }
}
