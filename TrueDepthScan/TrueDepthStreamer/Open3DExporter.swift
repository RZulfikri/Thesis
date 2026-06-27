//
//  Open3DExporter.swift
//  TrueDepthStreamer
//
//  Exports depth and video frames in Open3D TrueDepth registration format
//

import Foundation
import CoreVideo
import AVFoundation
import Accelerate
import os.log

struct ScanMetadata: Codable {
    let frameCount: Int
    let width: Int
    let height: Int
    let depthMinMeters: Double
    let depthMaxMeters: Double
    let frameDecimation: Int
    let videoChannels: Int
    let exportTimestamp: String
    let purpose: String
    let label: String
    let handedness: String  // "right", "left", atau "unknown"

    // Memberwise initialiser (required because we also define a custom Decodable init).
    init(frameCount: Int, width: Int, height: Int, depthMinMeters: Double, depthMaxMeters: Double,
         frameDecimation: Int, videoChannels: Int, exportTimestamp: String, purpose: String,
         label: String, handedness: String) {
        self.frameCount      = frameCount
        self.width           = width
        self.height          = height
        self.depthMinMeters  = depthMinMeters
        self.depthMaxMeters  = depthMaxMeters
        self.frameDecimation = frameDecimation
        self.videoChannels   = videoChannels
        self.exportTimestamp = exportTimestamp
        self.purpose         = purpose
        self.label           = label
        self.handedness      = handedness
    }

    // Custom decoder — field lama tanpa "label"/"handedness" tetap bisa dibaca.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        frameCount      = try c.decode(Int.self,    forKey: .frameCount)
        width           = try c.decode(Int.self,    forKey: .width)
        height          = try c.decode(Int.self,    forKey: .height)
        depthMinMeters  = try c.decode(Double.self, forKey: .depthMinMeters)
        depthMaxMeters  = try c.decode(Double.self, forKey: .depthMaxMeters)
        frameDecimation = try c.decode(Int.self,    forKey: .frameDecimation)
        videoChannels   = try c.decode(Int.self,    forKey: .videoChannels)
        exportTimestamp = try c.decode(String.self, forKey: .exportTimestamp)
        purpose         = try c.decode(String.self, forKey: .purpose)
        label           = try c.decodeIfPresent(String.self, forKey: .label)      ?? "Unknown"
        handedness      = try c.decodeIfPresent(String.self, forKey: .handedness) ?? "unknown"
    }
}

struct CameraCalibration: Codable {
    let width: Int
    let height: Int
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    let lensDistortionLookup: String?  // Base64 encoded float array
    let inverseLensDistortionLookup: String?  // Base64 encoded float array
    let lensDistortionCenter: [Double]?  // [x, y] center point
    let pixelSize: Double?  // Pixel size in mm (if available)
}

class Open3DExporter {

    private static let minExportDepth: Float = 0.10
    private static let maxExportDepth: Float = 0.50  // 50 cm — palm scanning range
    private static let log = OSLog(subsystem: Bundle.main.bundleIdentifier ?? "TrueDepthStreamer", category: "Open3DExporter")

    // MARK: - Export Main Function
    
    static func exportFrames(
        depthFrames: [CVPixelBuffer],
        roiPerFrame: [CGRect] = [],
        landmarksPerFrame: [[CGPoint]] = [],
        calibrationData: AVCameraCalibrationData?,
        outputFolderName: String,
        frameDecimation: Int = 5,
        label: String = "unknown",
        handedness: String = "unknown"
    ) -> URL? {

        guard depthFrames.count > 0 else {
            os_log(.error, log: log, "❌ No frames to export")
            return nil
        }

        // Create output directory: Documents/[label]/[label_timestamp]/
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let labelURL  = documentsPath.appendingPathComponent(label)
        let outputURL = labelURL.appendingPathComponent(outputFolderName)

        do {
            // Remove existing scan folder if it exists (label folder tetap)
            if FileManager.default.fileExists(atPath: outputURL.path) {
                try FileManager.default.removeItem(at: outputURL)
            }
            try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
        } catch {
            os_log(.error, log: log, "❌ Failed to create output directory: %{public}@", error.localizedDescription)
            return nil
        }

        os_log(.info, log: log, "📁 Exporting to: %{public}@", outputURL.path)

        // Export calibration.json
        if !exportCalibration(calibrationData: calibrationData, outputURL: outputURL) {
            return nil
        }

        // Export depth frames (depth-only, no video)
        for (index, depthBuffer) in depthFrames.enumerated() {
            os_log(.debug, log: log, "📝 Exporting depth frame %d/%d", index + 1, depthFrames.count)

            let roi       = index < roiPerFrame.count      ? roiPerFrame[index]      : .zero
            let landmarks = index < landmarksPerFrame.count ? landmarksPerFrame[index] : []
            let depthFilename = String(format: "depth%02d.bin", index)
            if !exportDepthFrame(depthBuffer, roi: roi, landmarks: landmarks, to: outputURL.appendingPathComponent(depthFilename)) {
                os_log(.error, log: log, "❌ Failed to export depth frame %d", index)
                return nil
            }
        }

        // Export metadata.json
        let width = CVPixelBufferGetWidth(depthFrames[0])
        let height = CVPixelBufferGetHeight(depthFrames[0])
        exportMetadata(
            frameCount: depthFrames.count,
            width: width,
            height: height,
            frameDecimation: frameDecimation,
            label: label,
            handedness: handedness,
            outputURL: outputURL
        )

        os_log(.info, log: log, "✅ Successfully exported %d depth frames to Open3D format", depthFrames.count)
        return outputURL
    }
    
    // MARK: - Metadata Export

    private static func exportMetadata(frameCount: Int, width: Int, height: Int, frameDecimation: Int,
                                       label: String, handedness: String, outputURL: URL) {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
        let metadata = ScanMetadata(
            frameCount: frameCount,
            width: width,
            height: height,
            depthMinMeters: 0.1,
            depthMaxMeters: 0.5,
            frameDecimation: frameDecimation,
            videoChannels: 0,  // depth-only export
            exportTimestamp: formatter.string(from: Date()),
            purpose: "3d-cnn-palm-recognition",
            label: label,
            handedness: handedness
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        if let data = try? encoder.encode(metadata) {
            try? data.write(to: outputURL.appendingPathComponent("metadata.json"))
            print("✅ Exported metadata.json (\(frameCount) frames, \(width)x\(height))")
        }
    }

    // MARK: - Calibration Export
    
    private static func exportCalibration(calibrationData: AVCameraCalibrationData?, outputURL: URL) -> Bool {
        guard let calibrationData = calibrationData else {
            os_log(.error, log: log, "❌ No calibration data available")
            return false
        }
        
        let intrinsicMatrix = calibrationData.intrinsicMatrix
        let referenceDimensions = calibrationData.intrinsicMatrixReferenceDimensions
        
        let fx = Double(intrinsicMatrix[0][0])
        let fy = Double(intrinsicMatrix[1][1])
        let cx = Double(intrinsicMatrix[2][0])
        let cy = Double(intrinsicMatrix[2][1])
        
        // Extract lens distortion lookup tables
        var lensDistortionLookupBase64: String?
        var inverseLensDistortionLookupBase64: String?
        var lensDistortionCenter: [Double]?
        var pixelSize: Double?
        
        // Get lens distortion lookup table
        if let lensDistortionLookupTable = calibrationData.lensDistortionLookupTable {
            lensDistortionLookupBase64 = lensDistortionLookupTable.base64EncodedString()
            let count = lensDistortionLookupTable.count / MemoryLayout<Float>.size
            os_log(.info, log: log, "✅ Extracted lens distortion lookup table (%d values)", count)
        }

        // Get inverse lens distortion lookup table
        if let inverseLensDistortionLookupTable = calibrationData.inverseLensDistortionLookupTable {
            inverseLensDistortionLookupBase64 = inverseLensDistortionLookupTable.base64EncodedString()
            let count = inverseLensDistortionLookupTable.count / MemoryLayout<Float>.size
            os_log(.info, log: log, "✅ Extracted inverse lens distortion lookup table (%d values)", count)
        }

        // Get lens distortion center
        let center = calibrationData.lensDistortionCenter
        lensDistortionCenter = [Double(center.x), Double(center.y)]
        os_log(.info, log: log, "✅ Lens distortion center: [%f, %f]", center.x, center.y)
        
        
        // Get pixel size (if available)
        if #available(iOS 14.0, *) {
            pixelSize = Double(calibrationData.pixelSize)
            os_log(.info, log: log, "✅ Pixel size: %f mm", pixelSize ?? 0)
        }
        let calibration = CameraCalibration(
            width: Int(referenceDimensions.width),
            height: Int(referenceDimensions.height),
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            lensDistortionLookup: lensDistortionLookupBase64,
            inverseLensDistortionLookup: inverseLensDistortionLookupBase64,
            lensDistortionCenter: lensDistortionCenter,
            pixelSize: pixelSize
        )
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        do {
            let jsonData = try encoder.encode(calibration)
            let calibrationURL = outputURL.appendingPathComponent("calibration.json")
            try jsonData.write(to: calibrationURL)
            os_log(.info, log: log, "✅ Exported calibration.json with lens distortion data")
            return true
        } catch {
            os_log(.error, log: log, "❌ Failed to export calibration: %{public}@", error.localizedDescription)
            return false
        }
    }
    
    
    // MARK: - Depth Frame Export
    
    /// Builds a 1-byte-per-pixel hand mask from Vision landmark points using CoreGraphics.
    ///
    /// Vision portrait coords (y-up, origin bottom-left) are converted to the landscape
    /// depth buffer space:  bufX = (1 - vy) * width,  bufY = (1 - vx) * height
    /// The convex hull of the converted points is expanded 15 px outward from the centroid
    /// to ensure palm edges and fingertips are not accidentally clipped.
    ///
    /// Returns a `[UInt8]` of size `width * height` where 255 = inside hand, 0 = outside.
    /// Returns nil when landmarks is empty or the hull has fewer than 3 vertices.
    private static func buildHandMask(landmarks: [CGPoint], width: Int, height: Int) -> [UInt8]? {
        guard landmarks.count >= 3 else { return nil }

        // Convert Vision portrait → buffer landscape pixel coords
        let bufPts = landmarks.map { p in
            CGPoint(x: (1.0 - p.y) * CGFloat(width),
                    y: (1.0 - p.x) * CGFloat(height))
        }

        // Compute convex hull (gift-wrapping, O(nh) for ≤16 points)
        var startIdx = 0
        for i in 1..<bufPts.count {
            if bufPts[i].x < bufPts[startIdx].x { startIdx = i }
        }
        var hull: [CGPoint] = []
        var cur = startIdx
        repeat {
            hull.append(bufPts[cur])
            var nxt = (cur + 1) % bufPts.count
            for i in 0..<bufPts.count {
                let o = bufPts[cur], a = bufPts[nxt], b = bufPts[i]
                if (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x) < 0 { nxt = i }
            }
            cur = nxt
        } while cur != startIdx && hull.count <= bufPts.count + 1
        guard hull.count >= 3 else { return nil }

        // Expand hull 15 px outward from centroid
        let cx = hull.map(\.x).reduce(0, +) / CGFloat(hull.count)
        let cy = hull.map(\.y).reduce(0, +) / CGFloat(hull.count)
        let expanded = hull.map { p -> CGPoint in
            let dx = p.x - cx, dy = p.y - cy
            let d = sqrt(dx * dx + dy * dy)
            guard d > 1 else { return p }
            return CGPoint(x: p.x + dx / d * 30, y: p.y + dy / d * 30)
        }

        // Rasterise polygon into a grayscale bitmap via CoreGraphics
        var mask = [UInt8](repeating: 0, count: width * height)
        mask.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else { return }
            ctx.setFillColor(gray: 0, alpha: 1)
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            ctx.setFillColor(gray: 1, alpha: 1)
            ctx.beginPath()
            ctx.move(to: expanded[0])
            expanded.dropFirst().forEach { ctx.addLine(to: $0) }
            ctx.closePath()
            ctx.fillPath()
        }
        return mask
    }

    private static func exportDepthFrame(_ pixelBuffer: CVPixelBuffer, roi: CGRect, landmarks: [CGPoint], to url: URL) -> Bool {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            os_log(.error, log: log, "❌ Failed to get depth buffer base address")
            return false
        }

        // --- Spatial mask ---
        // Priority: polygon mask from hand landmarks (precise) > bounding-box ROI (coarse) > none.
        //
        // Vision coords (portrait, y-up, origin bottom-left) → buffer (landscapeLeft):
        //   bufX = (1 - vy) * width,   bufY = (1 - vx) * height
        let polygonMask: [UInt8]? = buildHandMask(landmarks: landmarks, width: width, height: height)

        // Bounding-box fallback (only used when no polygon mask is available)
        let hasROI = polygonMask == nil && roi.width > 0 && roi.height > 0
        var bxMin = 0, bxMax = width - 1
        var byMin = 0, byMax = height - 1
        if hasROI {
            let padX = roi.width * 0.25
            let padY = roi.height * 0.25
            let minX = max(0.0, roi.minX - padX)
            let maxX = min(1.0, roi.maxX + padX)
            let minY = max(0.0, roi.minY - padY)
            let maxY = min(1.0, roi.maxY + padY)
            bxMin = max(0,          Int((1.0 - maxY) * Double(width)))
            bxMax = min(width - 1,  Int((1.0 - minY) * Double(width)))
            byMin = max(0,          Int((1.0 - maxX) * Double(height)))
            byMax = min(height - 1, Int((1.0 - minX) * Double(height)))
        }

        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)

        // Build a Float32 grid first so we can apply the neighborhood density filter.
        var grid = [Float32](repeating: 0, count: width * height)

        /// Returns true when a pixel at (x, y) is within the active spatial mask.
        func inMask(_ x: Int, _ y: Int) -> Bool {
            if let pm = polygonMask { return pm[y * width + x] > 0 }
            if hasROI { return x >= bxMin && x <= bxMax && y >= byMin && y <= byMax }
            return true  // no mask — keep all pixels
        }

        if pixelFormat == kCVPixelFormatType_DepthFloat32 {
            let floatPtr = baseAddress.assumingMemoryBound(to: Float32.self)
            for y in 0..<height {
                let rowPtr = floatPtr.advanced(by: y * bytesPerRow / MemoryLayout<Float32>.size)
                for x in 0..<width {
                    let depth = rowPtr[x]
                    grid[y * width + x] = (depth.isNaN || depth > maxExportDepth || depth < minExportDepth || !inMask(x, y)) ? 0.0 : depth
                }
            }
        } else if pixelFormat == kCVPixelFormatType_DepthFloat16 {
            // Use Accelerate vImage for correct IEEE 754 Float16 → Float32 conversion
            let float16Ptr = baseAddress.assumingMemoryBound(to: UInt16.self)
            var rowFloat32 = [Float32](repeating: 0, count: width)
            for y in 0..<height {
                let rowPtr = float16Ptr.advanced(by: y * bytesPerRow / MemoryLayout<UInt16>.size)
                rowFloat32.withUnsafeMutableBufferPointer { dstBuf in
                    var srcBuf = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: rowPtr),
                        height: 1,
                        width: vImagePixelCount(width),
                        rowBytes: width * MemoryLayout<UInt16>.size
                    )
                    var dstVBuf = vImage_Buffer(
                        data: dstBuf.baseAddress!,
                        height: 1,
                        width: vImagePixelCount(width),
                        rowBytes: width * MemoryLayout<Float32>.size
                    )
                    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstVBuf, 0)
                }
                for x in 0..<width {
                    let f = rowFloat32[x]
                    grid[y * width + x] = (f.isNaN || f > maxExportDepth || f < minExportDepth || !inMask(x, y)) ? 0.0 : f
                }
            }
        } else {
            os_log(.error, log: log, "❌ Unsupported depth format: %u", pixelFormat)
            return false
        }

        // Neighborhood density filter: remove isolated depth pixels.
        // A pixel is kept only if at least 3 of its 8-neighbors have a valid depth
        // value within ±50 mm. This eliminates stray background pixels that survive
        // the bounding-box ROI filter but are not part of the continuous palm surface.
        let neighborTolerance: Float32 = 0.05  // ±50 mm
        let minNeighbors = 3
        let dx = [-1, 0, 1, -1, 1, -1, 0, 1]
        let dy = [-1, -1, -1, 0, 0, 1, 1, 1]
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                let center = grid[idx]
                guard center > 0 else { continue }
                var validNeighbors = 0
                for k in 0..<8 {
                    let nx = x + dx[k], ny = y + dy[k]
                    guard nx >= 0 && nx < width && ny >= 0 && ny < height else { continue }
                    let nv = grid[ny * width + nx]
                    if nv > 0 && abs(nv - center) <= neighborTolerance { validNeighbors += 1 }
                }
                if validNeighbors < minNeighbors { grid[idx] = 0 }
            }
        }

        var depthData = Data()
        depthData.reserveCapacity(width * height * MemoryLayout<Float32>.size)
        for i in 0..<(width * height) {
            var v = grid[i]
            depthData.append(Data(bytes: &v, count: MemoryLayout<Float32>.size))
        }

        do {
            try depthData.write(to: url)
            return true
        } catch {
            os_log(.error, log: log, "❌ Failed to write depth file: %{public}@", error.localizedDescription)
            return false
        }
    }
    
}

