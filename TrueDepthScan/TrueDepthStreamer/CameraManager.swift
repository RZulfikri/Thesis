/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Manages the camera session and data processing via AVFoundation for continuous depth data.
*/

import Foundation
import AVFoundation
import CoreVideo
import UIKit
import Accelerate
import VideoToolbox
import QuartzCore
import Vision
import os.log

// Camera intrinsics for Open3D export
struct CameraIntrinsics {
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    let referenceDimensions: CGSize
}

// MARK: - Voxel Grid Types (2 mm resolution for temporal deduplication)
private struct VoxelKey: Hashable {
    let xi: Int32
    let yi: Int32
    let zi: Int32
}

private struct VoxelEntry {
    var sumX: Float
    var sumY: Float
    var sumZ: Float
    var count: UInt32
}

private let cameraLog = OSLog(subsystem: Bundle.main.bundleIdentifier ?? "TrueDepthStreamer", category: "CameraManager")

class CameraManager: NSObject, ObservableObject {

    enum SessionSetupResult {
        case success
        case notAuthorized
        case configurationFailed
    }

    enum DepthState: Equatable {
        case noObject   // nothing in scan range
        case tooClose   // object < min depth
        case tooFar     // object > max depth
        case inRange    // object at 10-50cm — ready to scan
    }

    @Published var permissionGranted = false
    @Published var sessionRunning = false
    @Published var isJetEnabled = false
    // Depth smoothing is always OFF — raw depth preserves palm surface detail for CNN.
    // Mix factor 0.35 = 65% camera + 35% JET depth overlay.
    // Shows camera feed with a colored depth heatmap so user can see what's being scanned.
    private let depthMixFactor: Float = 0.35

    // Outputs for UI
    @Published var previewPixelBuffer: CVPixelBuffer?
    @Published var capturedDepthData: AVDepthData?
    @Published var capturedVideoTexture: CVPixelBuffer?
    @Published var plyFileURL: URL?
    @Published var isScanning = false
    @Published var isUploading = false
    @Published var isProcessing = false

    @Published var touchDepth: String = ""
    @Published var thermalState: ProcessInfo.ThermalState = .nominal
    @Published var alertMessage: AlertMessage?
    @Published var trackingStateMessage: String = "Initializing..."
    @Published var lastExportedOpen3DURL: URL?
    @Published var isOnline: Bool = false
    @Published var shouldShowHistory: Bool = false

    // Device orientation for Vision hand pose (updated from main thread)
    private var currentInterfaceOrientation: UIInterfaceOrientation = .portrait

    // Depth-based scan readiness
    @Published var depthState: DepthState = .noObject
    /// Bounding box of the detected palm in Vision normalized coords (for ROI export masking).
    @Published var palmROI: CGRect = .zero

    struct AlertMessage: Identifiable {
        let id = UUID()
        let title: String
        let message: String
    }

    // AVFoundation Session
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session queue", attributes: [], autoreleaseFrequency: .workItem)
    private var setupResult: SessionSetupResult = .success

    private let dataOutputQueue = DispatchQueue(label: "video data queue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)

    private let videoDepthMixer = VideoMixer()

    // Open3D Export: Store depth frames for reconstruction (depth-only, no video needed)
    private var capturedDepthFrames: [CVPixelBuffer] = []
    /// Per-frame palm ROI in Vision normalized portrait coords (y-up). .zero = no palm detected.
    private var capturedDepthROIs: [CGRect] = []
    /// Per-frame hand landmark points in Vision normalized portrait coords (y-up, origin bottom-left).
    /// Used to build a precise polygon mask per frame during export.
    private var capturedDepthLandmarks: [[CGPoint]] = []
    /// Per-frame chirality votes ("right" / "left") untuk majority voting saat export.
    private var capturedHandednessVotes: [String] = []
    private let maxFramesToCapture = 10
    private var frameDecimation = 3
    private let minAccumulationDepth: Float = 0.10
    private let maxAccumulationDepth: Float = 0.60
    private let videoDepthConverter = DepthToJETConverter()

    private var renderingEnabled = true
    var maxRecordingDuration: TimeInterval = 5.0  // safety timeout — scan berhenti via frame count, bukan duration
    private var scanStartMonotonic: CFTimeInterval = 0
    private var appendedFramesThisCycle: Int = 0

    // Capture outputs
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let depthDataOutput = AVCaptureDepthDataOutput()
    private var outputSynchronizer: AVCaptureDataOutputSynchronizer?

    // Camera intrinsics
    private var cameraIntrinsics: matrix_float3x3?
    private var referenceDimensions: CGSize?
    private var cameraCalibrationData: AVCameraCalibrationData?

    // Point Cloud Accumulation (voxel grid for temporal deduplication at 2 mm resolution)
    private var voxelGrid: [VoxelKey: VoxelEntry] = [:]
    private let voxelSize: Float = 0.002  // 2 mm voxel resolution
    private var isAccumulating = false
    private var frameCounter = 0  // For frame decimation
    private let accumulationInterval = 0.1 // Seconds between accumulations
    private var lastAccumulationTime: TimeInterval = 0
    private var frameTimestamp: TimeInterval = 0

    // Object Tracking for Continuous Multi-Angle Scanning
    private var minObservedDepth: Float = Float.greatestFiniteMagnitude
    private var maxObservedDepth: Float = 0.0
    private var previousClosestDepth: Float = 0.0
    private let maxAllowedDepthRange: Float = 0.15  // 15cm
    private var maxFrameDepthChange: Float = 0.10  // 10cm
    private let spatialDepthTolerance: Float = 0.03  // 3cm

    // Palm / hand detection
    private var palmMissingFrames: Int = 0
    /// Latest hand landmark points in Vision normalized portrait coords (y-up, origin bottom-left).
    /// Used by processPreview to mask the JET depth overlay to the hand region only.
    private var lastHandLandmarks: [CGPoint] = []
    /// ~0.3 seconds at 30 fps before scan is halted when palm disappears or orientation changes
    private let palmMissingFrameThreshold: Int = 10
    /// Debounce: only publish depth state after consistent for N frames
    private var rawDepthStateBuffer: [DepthState] = []
    private let depthStateDebounceFrames: Int = 4  // ~0.13s at 30fps

    @available(iOS 14.0, *)
    private lazy var handPoseRequest: VNDetectHumanHandPoseRequest = {
        let req = VNDetectHumanHandPoseRequest()
        req.maximumHandCount = 1
        return req
    }()

    // Touch handling for depth display
    var textureTransformResolver: ((CGPoint) -> CGPoint?)?
    private var touchCoordinates: CGPoint = .zero
    private var touchDetected = false

    override init() {
        super.init()

        // Check video authorization
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            break
        case .notDetermined:
            sessionQueue.suspend()
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if !granted {
                    self.setupResult = .notAuthorized
                }
                self.sessionQueue.resume()
            }
        default:
            setupResult = .notAuthorized
        }

        addObservers()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        captureSession.stopRunning()
    }

    // MARK: - Session Management

    func startSession() {
        sessionQueue.async {
            self.configureSession()

            if self.setupResult == .success {
                DispatchQueue.main.async {
                    self.permissionGranted = true
                }
            }
        }
    }

    private func configureSession() {
        guard setupResult == .success else { return }

        captureSession.beginConfiguration()
        // .inputPriority lets us set activeFormat manually for maximum depth resolution
        captureSession.sessionPreset = .inputPriority

        // Get TrueDepth camera
        guard let videoDevice = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            os_log(.error, log: cameraLog, "❌ TrueDepth camera not available")
            setupResult = .configurationFailed
            captureSession.commitConfiguration()
            return
        }

        // Add video input
        do {
            let videoDeviceInput = try AVCaptureDeviceInput(device: videoDevice)

            if captureSession.canAddInput(videoDeviceInput) {
                captureSession.addInput(videoDeviceInput)
            } else {
                os_log(.error, log: cameraLog, "❌ Couldn't add video device input")
                setupResult = .configurationFailed
                captureSession.commitConfiguration()
                return
            }
        } catch {
            os_log(.error, log: cameraLog, "❌ Couldn't create video device input: %{public}@", error.localizedDescription)
            setupResult = .configurationFailed
            captureSession.commitConfiguration()
            return
        }

        // Select the highest-resolution format that supports depth data delivery.
        // On iPhone 14 TrueDepth this is typically 1280×960 or 1920×1440 vs the
        // default session preset which caps at 640×480.
        selectBestDepthFormat(for: videoDevice)

        // Configure video output
        videoDataOutput.alwaysDiscardsLateVideoFrames = true

        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        } else {
            os_log(.error, log: cameraLog, "❌ Couldn't add video data output")
            setupResult = .configurationFailed
            captureSession.commitConfiguration()
            return
        }

        // Configure depth output
        depthDataOutput.isFilteringEnabled = false  // Raw depth preserves surface detail for texture analysis
        depthDataOutput.alwaysDiscardsLateDepthData = true

        if captureSession.canAddOutput(depthDataOutput) {
            captureSession.addOutput(depthDataOutput)
        } else {
            os_log(.error, log: cameraLog, "❌ Couldn't add depth data output")
            setupResult = .configurationFailed
            captureSession.commitConfiguration()
            return
        }

        // Create output synchronizer
        outputSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoDataOutput, depthDataOutput])
        outputSynchronizer!.setDelegate(self, queue: dataOutputQueue)

        os_log(.info, log: cameraLog, "✅ Depth output configured - intrinsics will be extracted from depth data")

        captureSession.commitConfiguration()
        captureSession.startRunning()

        DispatchQueue.main.async {
            self.sessionRunning = self.captureSession.isRunning
            self.trackingStateMessage = "Ready"
            os_log(.info, log: cameraLog, "✅ AVCaptureSession started with TrueDepth camera")
        }
    }

    /// Enumerates the TrueDepth device formats and activates the one with the highest
    /// resolution that supports depth data delivery, preferring 30 fps.
    private func selectBestDepthFormat(for device: AVCaptureDevice) {
        // Score: width * 10000 + fps (prefer higher res, then higher fps up to 30)
        var bestFormat: AVCaptureDevice.Format? = nil
        var bestScore = 0

        for format in device.formats {
            guard !format.supportedDepthDataFormats.isEmpty else { continue }

            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let maxFps = format.videoSupportedFrameRateRanges
                .map { Int($0.maxFrameRate) }
                .filter { $0 <= 30 }
                .max() ?? 0
            guard maxFps > 0 else { continue }

            let score = Int(dims.width) * 10000 + maxFps
            if score > bestScore {
                bestScore = score
                bestFormat = format
            }
        }

        guard let format = bestFormat else {
            os_log(.info, log: cameraLog, "⚠️ No depth-capable format found — keeping default")
            return
        }

        do {
            try device.lockForConfiguration()
            device.activeFormat = format
            // Cap frame rate at 30 fps
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            device.unlockForConfiguration()

            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            os_log(.info, log: cameraLog, "✅ Depth format selected: %dx%d (depth formats: %d)",
                   dims.width, dims.height, format.supportedDepthDataFormats.count)
        } catch {
            os_log(.error, log: cameraLog, "❌ Failed to set device format: %{public}@", error.localizedDescription)
        }
    }

    func restartSession() {
        sessionQueue.async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
            self.configureSession()
        }
    }

    func stopSession() {
        sessionQueue.async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
                DispatchQueue.main.async {
                    self.sessionRunning = false
                }
            }
        }
    }

    // MARK: - Scanning Control

    func startCountdown() {
        guard depthState == .inRange && !isScanning else { return }
        startScanning()
    }

    private func startScanning() {
        voxelGrid.removeAll()
        clearCapturedFrames()
        palmMissingFrames = 0


        DispatchQueue.main.async {
            self.lastExportedOpen3DURL = nil
        }

        minObservedDepth = Float.greatestFiniteMagnitude
        maxObservedDepth = 0.0
        previousClosestDepth = 0.0

        isScanning = true
        isAccumulating = true
        scanStartMonotonic = CACurrentMediaTime()
    }

    func stopScanning() {
        if isScanning {
            isScanning = false
            isAccumulating = false
            scanStartMonotonic = 0

            // Snapshot mutable state on dataOutputQueue for thread safety before handing off
            dataOutputQueue.async {
                let (pts, cols) = self.getVoxelizedPoints()
                let depthFrames = self.capturedDepthFrames
                let depthROIs   = self.capturedDepthROIs
                let calibData   = self.cameraCalibrationData

                // Majority vote dari chirality yang terdeteksi per frame
                let votes = self.capturedHandednessVotes
                let handedness: String = {
                    let rights = votes.filter { $0 == "right" }.count
                    let lefts  = votes.filter { $0 == "left"  }.count
                    if rights == 0 && lefts == 0 { return "unknown" }
                    return rights >= lefts ? "right" : "left"
                }()
                os_log(.info, log: cameraLog, "🖐 Handedness: %{public}@ (right=%d left=%d)", handedness, votes.filter { $0 == "right" }.count, votes.filter { $0 == "left" }.count)

                let dateFormatter = DateFormatter()
                dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
                let timestamp = dateFormatter.string(from: Date())
                let label     = self.currentLabel()
                let folderName = "\(label)_\(timestamp)"

                let depthLandmarks = self.capturedDepthLandmarks

                let capturedData = CapturedScanData(
                    label:           label,
                    timestamp:       timestamp,
                    folderName:      folderName,
                    points:          pts,
                    colors:          cols,
                    depthFrames:     depthFrames,
                    depthROIs:       depthROIs,
                    depthLandmarks:  depthLandmarks,
                    calibrationData: calibData,
                    handedness:      handedness
                )

                // Serahkan ke background queue — UI tidak perlu menunggu
                DispatchQueue.main.async {
                    ProcessingQueue.shared.enqueue(capturedData)
                    self.isProcessing = false
                    self.clearCapturedFrames()
                }
            }
        }
    }

    func resetScan() {
        voxelGrid.removeAll()
        plyFileURL = nil

        minObservedDepth = Float.greatestFiniteMagnitude
        maxObservedDepth = 0.0
        previousClosestDepth = 0.0
    }

    // MARK: - Observers

    private func addObservers() {
        NotificationCenter.default.addObserver(self, selector: #selector(sessionRuntimeError), name: .AVCaptureSessionRuntimeError, object: captureSession)
        NotificationCenter.default.addObserver(self, selector: #selector(sessionWasInterrupted), name: .AVCaptureSessionWasInterrupted, object: captureSession)
        NotificationCenter.default.addObserver(self, selector: #selector(sessionInterruptionEnded), name: .AVCaptureSessionInterruptionEnded, object: captureSession)
    }

    @objc func sessionRuntimeError(notification: NSNotification) {
        guard let error = notification.userInfo?[AVCaptureSessionErrorKey] as? AVError else { return }
        os_log(.error, log: cameraLog, "❌ Capture session runtime error: %{public}@", error.localizedDescription)
    }

    @objc func sessionWasInterrupted(notification: NSNotification) {
        DispatchQueue.main.async {
            self.sessionRunning = false
        }
    }

    @objc func sessionInterruptionEnded(notification: NSNotification) {
        DispatchQueue.main.async {
            self.sessionRunning = self.captureSession.isRunning
        }
    }

    // MARK: - Touch Depth

    func getDepthAtPoint(_ point: CGPoint, depthFrame: CVPixelBuffer, textureSize: CGSize) {
        let width = CVPixelBufferGetWidth(depthFrame)
        let height = CVPixelBufferGetHeight(depthFrame)

        let x = Int(point.x / textureSize.width * CGFloat(width))
        let y = Int(point.y / textureSize.height * CGFloat(height))

        let clampedX = max(0, min(x, width - 1))
        let clampedY = max(0, min(y, height - 1))

        CVPixelBufferLockBaseAddress(depthFrame, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthFrame, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(depthFrame) else { return }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthFrame)
        // Depth buffer is always Float32 (converted in dataOutputSynchronizer)
        let rowData = baseAddress.advanced(by: clampedY * bytesPerRow)
        let f32Pixel = rowData.assumingMemoryBound(to: Float32.self)[clampedX]

        let depthString = String(format: "%.2f cm", f32Pixel * 100.0)
        DispatchQueue.main.async {
            self.touchDepth = depthString
        }
    }

    func setTouchCoordinates(_ point: CGPoint, detected: Bool) {
        touchCoordinates = point
        touchDetected = detected
    }

    /// Adjusts focus and exposure to the tapped point.
    /// - Parameter normalizedPoint: Point in [0, 1] x [0, 1] texture coordinates.
    func focusAndExpose(at normalizedPoint: CGPoint) {
        sessionQueue.async {
            guard let device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else { return }
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                if device.isFocusPointOfInterestSupported {
                    device.focusPointOfInterest = normalizedPoint
                    device.focusMode = .autoFocus
                }
                if device.isExposurePointOfInterestSupported {
                    device.exposurePointOfInterest = normalizedPoint
                    device.exposureMode = .autoExpose
                }
            } catch {
                // Non-critical: ignore if device can't be locked for configuration
            }
        }
    }
}

// MARK: - AVCaptureDataOutputSynchronizerDelegate

extension CameraManager: AVCaptureDataOutputSynchronizerDelegate {

    /// Maps current interface orientation to CGImagePropertyOrientation for front camera.
    /// TrueDepth delivers buffers in landscapeLeft; Vision needs the EXIF tag to interpret correctly.
    private func visionOrientation(for interfaceOrientation: UIInterfaceOrientation) -> CGImagePropertyOrientation {
        switch interfaceOrientation {
        case .portrait:            return .leftMirrored   // rotate 90° + mirror for front cam
        case .landscapeRight:      return .upMirrored     // just mirror horizontally
        case .landscapeLeft:       return .downMirrored   // rotate 180° + mirror
        case .portraitUpsideDown:  return .rightMirrored  // rotate 270° + mirror
        default:                   return .leftMirrored
        }
    }

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                               didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        appendedFramesThisCycle = 0

        // Snapshot current interface orientation (UIKit state, thread-safe read)
        let orientation: UIInterfaceOrientation = {
            if Thread.isMainThread {
                let scene = UIApplication.shared.connectedScenes
                    .compactMap { $0 as? UIWindowScene }
                    .first(where: { $0.activationState == .foregroundActive })
                return scene?.interfaceOrientation ?? .portrait
            } else {
                return self.currentInterfaceOrientation
            }
        }()
        // Update stored orientation for next frame (non-blocking)
        DispatchQueue.main.async {
            let scene = UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .first(where: { $0.activationState == .foregroundActive })
            self.currentInterfaceOrientation = scene?.interfaceOrientation ?? .portrait
        }

        // Get synchronized depth and video data
        guard let syncedDepthData = synchronizedDataCollection.synchronizedData(for: depthDataOutput) as? AVCaptureSynchronizedDepthData,
              let syncedVideoData = synchronizedDataCollection.synchronizedData(for: videoDataOutput) as? AVCaptureSynchronizedSampleBufferData else {
            return
        }

        guard !syncedDepthData.depthDataWasDropped,
              !syncedVideoData.sampleBufferWasDropped else {
            return
        }

        let depthData = syncedDepthData.depthData

        // CRITICAL: Convert depth data to Float32 format
        // TrueDepth may provide disparity instead of depth
        let convertedDepthData: AVDepthData
        if depthData.depthDataType != kCVPixelFormatType_DepthFloat32 {
            convertedDepthData = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        } else {
            convertedDepthData = depthData
        }

        let depthPixelBuffer = convertedDepthData.depthDataMap

        // Extract camera intrinsics from depth data (first frame only)
        if cameraIntrinsics == nil, let calibrationData = convertedDepthData.cameraCalibrationData {
            cameraIntrinsics = calibrationData.intrinsicMatrix
            referenceDimensions = calibrationData.intrinsicMatrixReferenceDimensions
            cameraCalibrationData = calibrationData

            let fx = calibrationData.intrinsicMatrix[0][0]
            let fy = calibrationData.intrinsicMatrix[1][1]
            let cx = calibrationData.intrinsicMatrix[2][0]
            let cy = calibrationData.intrinsicMatrix[2][1]

            os_log(.info, log: cameraLog, "✅ Camera intrinsics: ref=%gx%g fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                   Double(calibrationData.intrinsicMatrixReferenceDimensions.width),
                   Double(calibrationData.intrinsicMatrixReferenceDimensions.height),
                   Double(fx), Double(fy), Double(cx), Double(cy))
        }

        guard let videoPixelBuffer = CMSampleBufferGetImageBuffer(syncedVideoData.sampleBuffer) else {
            return
        }

        // Update timestamp
        frameTimestamp = syncedDepthData.timestamp.seconds

        // --- Depth-based readiness check (no Vision dependency for trigger) ---
        let newDepthState = checkDepthState(depthBuffer: depthPixelBuffer)

        // Debounce depth state for UI stability
        rawDepthStateBuffer.append(newDepthState)
        if rawDepthStateBuffer.count > depthStateDebounceFrames {
            rawDepthStateBuffer.removeFirst(rawDepthStateBuffer.count - depthStateDebounceFrames)
        }
        let debouncedDepthState: DepthState
        if rawDepthStateBuffer.count >= depthStateDebounceFrames &&
           rawDepthStateBuffer.allSatisfy({ $0 == newDepthState }) {
            debouncedDepthState = newDepthState
        } else {
            debouncedDepthState = depthState  // keep previous to avoid flicker
        }

        // Vision hand pose — for ROI masking (export) and preview depth mask
        var detectedPalmROI: CGRect? = nil
        var detectedHandedness: String? = nil
        if #available(iOS 14.0, *) {
            let handResult = detectHandROI(in: videoPixelBuffer, orientation: visionOrientation(for: orientation))
            detectedPalmROI    = handResult?.roi
            detectedHandedness = handResult?.handedness
            lastHandLandmarks  = handResult?.landmarks ?? []
        }

        DispatchQueue.main.async {
            self.depthState = debouncedDepthState
            if let roi = detectedPalmROI { self.palmROI = roi }
        }

        // Stop scan if object leaves frame
        if isAccumulating {
            if newDepthState != .noObject {
                palmMissingFrames = 0
            } else {
                palmMissingFrames += 1
                if palmMissingFrames >= palmMissingFrameThreshold {
                    palmMissingFrames = 0
                    DispatchQueue.main.async {
                        self.alertMessage = AlertMessage(title: "Scan Stopped", message: "Object left the scan area.")
                        self.stopScanning()
                    }
                    return
                }
            }
        }


        // Safety timeout — cegah scan jalan selamanya jika frame sparse
        if isAccumulating && scanStartMonotonic > 0 && (CACurrentMediaTime() - scanStartMonotonic) >= maxRecordingDuration {
            DispatchQueue.main.async {
                self.stopScanning()
            }
            return
        }

        // Process preview for JET depth visualization
        processPreview(depthBuffer: depthPixelBuffer, videoBuffer: videoPixelBuffer)

        // Capture depth frames for Open3D export (with decimation, depth-only)
        if isAccumulating {
            frameCounter += 1
            if frameCounter % frameDecimation == 0 && capturedDepthFrames.count < maxFramesToCapture {
                if let depthCopy = D3ProcessingService.copyPixelBuffer(depthPixelBuffer) {
                    capturedDepthFrames.append(depthCopy)
                    capturedDepthROIs.append(detectedPalmROI ?? .zero)
                    capturedDepthLandmarks.append(lastHandLandmarks)
                    if let h = detectedHandedness { capturedHandednessVotes.append(h) }
                    appendedFramesThisCycle += 1
                    os_log(.debug, log: cameraLog, "📸 Captured depth frame %d/%d", capturedDepthFrames.count, maxFramesToCapture)

                    // Stop tepat saat target frame terpenuhi — tidak bergantung pada timing
                    if capturedDepthFrames.count >= maxFramesToCapture {
                        DispatchQueue.main.async { self.stopScanning() }
                    }
                }
            }
        }

        // Accumulate points into voxel grid
        if isAccumulating {
            if frameTimestamp - lastAccumulationTime > accumulationInterval {
                lastAccumulationTime = frameTimestamp
                accumulatePoints(depthBuffer: depthPixelBuffer, palmROI: detectedPalmROI ?? .zero)
            }
        }
    }
}

// MARK: - Depth State Check + Hand ROI

extension CameraManager {

    /// Samples the center region of the depth buffer to determine scan readiness.
    /// Uses a grid of depth samples — no Vision dependency.
    private func checkDepthState(depthBuffer: CVPixelBuffer) -> DepthState {
        let w = CVPixelBufferGetWidth(depthBuffer)
        let h = CVPixelBufferGetHeight(depthBuffer)

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depthBuffer) else { return .noObject }
        let rowBytes = CVPixelBufferGetBytesPerRow(depthBuffer)

        // Sample a 7×7 grid in the center 50% of the frame
        let startX = w / 4, endX = w * 3 / 4
        let startY = h / 4, endY = h * 3 / 4
        let stepX = max(1, (endX - startX) / 6)
        let stepY = max(1, (endY - startY) / 6)

        var inRangeCount = 0
        var tooCloseCount = 0
        var tooFarCount = 0
        var totalSamples = 0

        var py = startY
        while py <= endY {
            var px = startX
            while px <= endX {
                let ptr = base.advanced(by: py * rowBytes + px * MemoryLayout<Float32>.size)
                let d = ptr.assumingMemoryBound(to: Float32.self).pointee
                if d > 0 && !d.isNaN {
                    totalSamples += 1
                    if d < minAccumulationDepth {
                        tooCloseCount += 1
                    } else if d > maxAccumulationDepth {
                        tooFarCount += 1
                    } else {
                        inRangeCount += 1
                    }
                }
                px += stepX
            }
            py += stepY
        }

        // Need at least 5 valid samples in scan range to consider "in range"
        if inRangeCount >= 5 { return .inRange }
        if tooCloseCount > tooFarCount && tooCloseCount >= 3 { return .tooClose }
        if tooFarCount >= 3 { return .tooFar }
        return .noObject
    }

    /// Hasil deteksi tangan per frame: ROI bounding box + chirality + landmark points.
    struct HandDetectionResult {
        let roi: CGRect
        let handedness: String   // "right" atau "left"
        /// All detected joint locations in Vision normalized portrait coords (y-up, origin bottom-left).
        let landmarks: [CGPoint]
    }

    /// Runs Vision hand pose ONLY for ROI bounding box + chirality (used during export masking).
    /// Returns nil if no hand detected. Does NOT block scan trigger.
    @available(iOS 14.0, *)
    func detectHandROI(in pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation = .leftMirrored) -> HandDetectionResult? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                             orientation: orientation,
                                             options: [:])
        do {
            try handler.perform([handPoseRequest])
        } catch {
            return nil
        }

        guard let obs = handPoseRequest.results?.first else { return nil }

        // Collect all visible joints — full set including DIP joints for accurate
        // outer-edge coverage of thumb and pinky during polygon masking.
        let allJoints: [VNHumanHandPoseObservation.JointName] = [
            .wrist,
            .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
            .indexMCP, .indexPIP, .indexDIP, .indexTip,
            .middleMCP, .middlePIP, .middleDIP, .middleTip,
            .ringMCP, .ringPIP, .ringDIP, .ringTip,
            .littleMCP, .littlePIP, .littleDIP, .littleTip
        ]
        var pts: [CGPoint] = []
        for joint in allJoints {
            if let p = try? obs.recognizedPoint(joint), p.confidence > 0.3 {
                pts.append(p.location)
            }
        }
        guard pts.count >= 3 else { return nil }

        let xs = pts.map(\.x)
        let ys = pts.map(\.y)
        let pad: CGFloat = 0.06
        let roi = CGRect(
            x:      max(0, xs.min()! - pad),
            y:      max(0, ys.min()! - pad),
            width:  min(1, xs.max()! - xs.min()! + 2 * pad),
            height: min(1, ys.max()! - ys.min()! + 2 * pad)
        )

        // Chirality: iOS 15+ memberikan .left/.right langsung dari model Vision.
        // Pada iOS 14, fallback ke "unknown" karena API belum tersedia.
        //
        // PENTING — app ini selalu menggunakan kamera DEPAN (TrueDepth/selfie).
        // Kamera depan membalik gambar secara horizontal, sehingga Vision
        // mendeteksi chirality dari perspektif kamera — kebalikan dari pengguna.
        // Tangan KANAN pengguna muncul di sisi KIRI frame → Vision: .left → flip ke "right".
        var handedness = "unknown"
        if #available(iOS 15.0, *) {
            switch obs.chirality {
            case .right: handedness = "left"   // flip: kamera depan mirror
            case .left:  handedness = "right"  // flip: kamera depan mirror
            default:     handedness = "unknown"
            }
        }

        return HandDetectionResult(roi: roi, handedness: handedness, landmarks: pts)
    }
}

// MARK: - Point Cloud Accumulation

extension CameraManager {

    private func accumulatePoints(depthBuffer: CVPixelBuffer, palmROI: CGRect) {
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)

        guard let depthBase = CVPixelBufferGetBaseAddress(depthBuffer) else { return }

        let depthBytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)

        guard let intrinsics = cameraIntrinsics else {
            os_log(.error, log: cameraLog, "⚠️ Camera intrinsics not available")
            return
        }

        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        // Depth buffer is Float32 (already converted in dataOutputSynchronizer)
        let depthPtr = depthBase.assumingMemoryBound(to: Float32.self)

        // Find closest object using center 50% ROI to avoid edge objects hijacking focus detection
        let roiX0 = width / 4
        let roiX1 = 3 * width / 4
        let roiY0 = height / 4
        let roiY1 = 3 * height / 4
        var validDepths: [Float] = []
        let sampleStride = 4

        for y in stride(from: roiY0, to: roiY1, by: sampleStride) {
            let rowPtr = depthPtr.advanced(by: y * depthBytesPerRow / MemoryLayout<Float32>.size)
            for x in stride(from: roiX0, to: roiX1, by: sampleStride) {
                let depthFloat = rowPtr[x]
                if depthFloat > minAccumulationDepth && depthFloat < maxAccumulationDepth {
                    validDepths.append(depthFloat)
                }
            }
        }

        guard !validDepths.isEmpty else {
            os_log(.debug, log: cameraLog, "⚠️ No valid depth data in center ROI")
            return
        }

        let minDepth = validDepths.min()!
        let nearMinDepths = validDepths.filter { abs($0 - minDepth) < 0.02 }
        let currentClosestDepth = nearMinDepths.isEmpty ? minDepth : nearMinDepths.sorted()[nearMinDepths.count / 2]

        // Update adaptive JET preview range: ±10 cm window centered on the closest object
        videoDepthConverter.updateDepthRange(
            min: max(0.05, currentClosestDepth - 0.10),
            max: min(1.0, currentClosestDepth + 0.10)
        )

        // Update dynamic depth range for scan-stop detection
        if minObservedDepth == Float.greatestFiniteMagnitude {
            minObservedDepth = currentClosestDepth
            maxObservedDepth = currentClosestDepth
            previousClosestDepth = currentClosestDepth
            os_log(.info, log: cameraLog, "✅ Started scanning at: %.2f cm", Double(currentClosestDepth * 100))
        } else {
            let frameChange = abs(currentClosestDepth - previousClosestDepth)
            if frameChange > maxFrameDepthChange {
                os_log(.info, log: cameraLog, "✅ Scan complete - Object changed: %.1f cm sudden change", Double(frameChange * 100))
                if appendedFramesThisCycle > 0 && capturedDepthFrames.count >= appendedFramesThisCycle {
                    capturedDepthFrames.removeLast(appendedFramesThisCycle)
                }
                DispatchQueue.main.async { self.stopScanning() }
                return
            }

            minObservedDepth = min(minObservedDepth, currentClosestDepth)
            maxObservedDepth = max(maxObservedDepth, currentClosestDepth)

            let depthRange = maxObservedDepth - minObservedDepth
            if depthRange > maxAllowedDepthRange {
                os_log(.info, log: cameraLog, "✅ Scan complete - Object moved too far: %.1f cm range", Double(depthRange * 100))
                if appendedFramesThisCycle > 0 && capturedDepthFrames.count >= appendedFramesThisCycle {
                    capturedDepthFrames.removeLast(appendedFramesThisCycle)
                }
                DispatchQueue.main.async { self.stopScanning() }
                return
            }

            previousClosestDepth = currentClosestDepth
        }

        // Convert Vision-normalized palm ROI to depth-buffer pixel bounds.
        // Vision: portrait y-up (origin bottom-left). Buffer: landscape 640×480.
        //   bufferX = (1 - vision_y) * width
        //   bufferY = (1 - vision_x) * height
        let hasROI = palmROI.width > 0 && palmROI.height > 0
        var accX0 = 0, accX1 = width - 1
        var accY0 = 0, accY1 = height - 1
        if hasROI {
            let pad: CGFloat = 0.08  // 8% padding so fingers aren't clipped
            let minX = max(0.0, palmROI.minX - pad)
            let maxX = min(1.0, palmROI.maxX + pad)
            let minY = max(0.0, palmROI.minY - pad)
            let maxY = min(1.0, palmROI.maxY + pad)
            accX0 = max(0,          Int((1.0 - maxY) * Double(width)))
            accX1 = min(width - 1,  Int((1.0 - minY) * Double(width)))
            accY0 = max(0,          Int((1.0 - maxX) * Double(height)))
            accY1 = min(height - 1, Int((1.0 - minX) * Double(height)))
        }

        // Accumulate points into voxel grid — restricted to the palm ROI pixel region.
        // Depth buffer is Float32 (already converted in dataOutputSynchronizer)
        for y in stride(from: accY0, to: accY1 + 1, by: sampleStride) {
            let rowPtr = depthPtr.advanced(by: y * depthBytesPerRow / MemoryLayout<Float32>.size)
            for x in stride(from: accX0, to: accX1 + 1, by: sampleStride) {
                let depthFloat = rowPtr[x]

                guard abs(depthFloat - currentClosestDepth) < spatialDepthTolerance else { continue }

                let xW = (Float(x) - cx) * depthFloat / fx
                let yW = (Float(y) - cy) * depthFloat / fy
                let point = simd_float3(xW, -yW, -depthFloat)

                // Insert/merge into voxel grid — collapses temporal duplicates at 2 mm resolution
                let xi = Int32(floor(point.x / voxelSize))
                let yi = Int32(floor(point.y / voxelSize))
                let zi = Int32(floor(point.z / voxelSize))
                let key = VoxelKey(xi: xi, yi: yi, zi: zi)

                if var entry = voxelGrid[key] {
                    entry.sumX += point.x
                    entry.sumY += point.y
                    entry.sumZ += point.z
                    entry.count += 1
                    voxelGrid[key] = entry
                } else {
                    voxelGrid[key] = VoxelEntry(sumX: point.x, sumY: point.y, sumZ: point.z, count: 1)
                }
            }
        }

        let depthRange = maxObservedDepth - minObservedDepth
        os_log(.debug, log: cameraLog, "📊 Voxels: %d, Current: %.2f cm, Range: %.1f cm",
               voxelGrid.count, Double(currentClosestDepth * 100), Double(depthRange * 100))
    }

    /// Averages each voxel bucket to produce the final deduplicated point cloud.
    /// Must be called from dataOutputQueue for thread safety.
    private func getVoxelizedPoints() -> ([simd_float3], [simd_uchar3]) {
        var points = [simd_float3]()
        var colors = [simd_uchar3]()
        points.reserveCapacity(voxelGrid.count)
        colors.reserveCapacity(voxelGrid.count)
        for entry in voxelGrid.values {
            let c = Float(entry.count)
            points.append(simd_float3(entry.sumX / c, entry.sumY / c, entry.sumZ / c))
            colors.append(simd_uchar3(128, 128, 128))  // Neutral grey (depth-only scan)
        }
        return (points, colors)
    }
}

// MARK: - Preview Processing

extension CameraManager {

    private func processPreview(depthBuffer: CVPixelBuffer, videoBuffer: CVPixelBuffer) {
        // JET depth colormap (needed as second input to the mixer even when mixFactor = 0)
        if !videoDepthConverter.isPrepared {
            var depthFormatDescription: CMFormatDescription?
            CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                         imageBuffer: depthBuffer,
                                                         formatDescriptionOut: &depthFormatDescription)
            if let desc = depthFormatDescription {
                videoDepthConverter.prepare(with: desc, outputRetainedBufferCountHint: 2)
            } else {
                return
            }
        }

        guard let jetPixelBuffer = videoDepthConverter.render(pixelBuffer: depthBuffer) else { return }

        // Convert video YCbCr → BGRA
        var bgraVideoBuffer: CVPixelBuffer?
        if let converted = convertYCbCrToBGRA(videoBuffer) {
            bgraVideoBuffer = converted
        } else {
            bgraVideoBuffer = videoBuffer
        }
        guard let finalVideoBuffer = bgraVideoBuffer else { return }

        // Mix video + JET depth. With mixFactor = 0.0 the output is pure RGB camera.
        // The mixer output is IOSurface-backed which Metal's texture cache requires.
        if !videoDepthMixer.isPrepared {
            var videoFormatDescription: CMFormatDescription?
            CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                         imageBuffer: finalVideoBuffer,
                                                         formatDescriptionOut: &videoFormatDescription)
            if let desc = videoFormatDescription {
                videoDepthMixer.prepare(with: desc, outputRetainedBufferCountHint: 3)
                videoDepthMixer.mixFactor = depthMixFactor
            } else {
                return
            }
        }

        autoreleasepool {
            guard let mixedBuffer = videoDepthMixer.mix(videoPixelBuffer: finalVideoBuffer, depthPixelBuffer: jetPixelBuffer) else { return }
            DispatchQueue.main.async { self.previewPixelBuffer = mixedBuffer }
        }
    }

    private func convertYCbCrToBGRA(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? { D3ProcessingService.convertYCbCrToBGRA(pixelBuffer) }

    // MARK: - Open3D Export Functions

    func getCameraIntrinsics() -> CameraIntrinsics? {
        guard let intrinsics = cameraIntrinsics,
              let refDims = referenceDimensions else {
            return nil
        }

        let fx = Double(intrinsics[0][0])
        let fy = Double(intrinsics[1][1])
        let cx = Double(intrinsics[2][0])
        let cy = Double(intrinsics[2][1])

        return CameraIntrinsics(fx: fx, fy: fy, cx: cx, cy: cy, referenceDimensions: refDims)
    }

    private func copyPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? { D3ProcessingService.copyPixelBuffer(pixelBuffer) }

    func exportForOpen3D() -> URL? {
        os_log(.info, log: cameraLog, "🔍 Export for Open3D called: depth=%d", capturedDepthFrames.count)

        guard let intrinsics = getCameraIntrinsics() else {
            os_log(.error, log: cameraLog, "❌ No camera intrinsics available")
            DispatchQueue.main.async {
                self.alertMessage = AlertMessage(
                    title: "Export Failed",
                    message: "Camera intrinsics not available. Please restart the app."
                )
            }
            return nil
        }

        os_log(.info, log: cameraLog, "   Intrinsics: fx=%g fy=%g", intrinsics.fx, intrinsics.fy)

        guard capturedDepthFrames.count > 0 else {
            os_log(.error, log: cameraLog, "❌ No frames captured for export")
            DispatchQueue.main.async {
                self.alertMessage = AlertMessage(
                    title: "Export Failed",
                    message: "No frames were captured during scanning. Please try scanning again."
                )
            }
            return nil
        }

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let folderName = "\(currentLabel())_\(timestamp)"

        os_log(.info, log: cameraLog, "📦 Exporting %d depth frames for Open3D...", capturedDepthFrames.count)

        let result = D3ProcessingService.exportForOpen3D(
            depthFrames: capturedDepthFrames,
            roiPerFrame: capturedDepthROIs,
            calibrationData: cameraCalibrationData,
            folderName: folderName,
            label: currentLabel()
        )

        if result == nil {
            DispatchQueue.main.async {
                self.alertMessage = AlertMessage(
                    title: "Export Failed",
                    message: "Failed to export frames. Check console for details."
                )
            }
        } else {
            os_log(.info, log: cameraLog, "✅ Export successful: %{public}@", result!.path)
            DispatchQueue.main.async {
                self.lastExportedOpen3DURL = result
            }
        }

        return result
    }

    func clearCapturedFrames() {
        capturedDepthFrames.removeAll()
        capturedDepthROIs.removeAll()
        capturedDepthLandmarks.removeAll()
        capturedHandednessVotes.removeAll()
        frameCounter = 0
    }

    // MARK: - Label Helper

    /// Returns the current scan label sanitised to alphanumeric + `-_` characters.
    func currentLabel() -> String {
        let rawLabel = UserDefaults.standard.string(forKey: "topLabelText") ?? "3d-label"
        let trimmed = rawLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        let filtered = String(trimmed.filter { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" })
        return filtered.isEmpty ? "unknown" : filtered
    }

    // MARK: - Upload Open3D data
    private func uploadOpen3D(from folderURL: URL) {
        let label = currentLabel()
        let uuid = UUID().uuidString
        os_log(.info, log: cameraLog, "Uploading with uuid=%{public}@, label=%{public}@", uuid, label)

        DispatchQueue.main.async {
            self.isUploading = true
        }

        ApiService.shared.uploadRegistrationOpen3D(folderURL: folderURL, uuid: uuid, label: label) { result in
            DispatchQueue.main.async {
                self.isUploading = false
                self.isProcessing = false
                self.clearCapturedFrames()
                switch result {
                case .success:
                    self.shouldShowHistory = true
                case .failure(let err):
                    self.alertMessage = AlertMessage(title: "Upload Error", message: err.localizedDescription)
                }
            }
        }
    }
}
