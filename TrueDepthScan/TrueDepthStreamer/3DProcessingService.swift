import Foundation
import CoreVideo
import AVFoundation
import CoreImage

final class D3ProcessingService {

    // MARK: - Open3D Export

    static func exportForOpen3D(
        depthFrames: [CVPixelBuffer],
        roiPerFrame: [CGRect] = [],
        landmarksPerFrame: [[CGPoint]] = [],
        calibrationData: AVCameraCalibrationData?,
        folderName: String,
        label: String = "unknown",
        handedness: String = "unknown"
    ) -> URL? {
        Open3DExporter.exportFrames(
            depthFrames:       depthFrames,
            roiPerFrame:       roiPerFrame,
            landmarksPerFrame: landmarksPerFrame,
            calibrationData:   calibrationData,
            outputFolderName:  folderName,
            label:             label,
            handedness:        handedness
        )
    }

    // MARK: - PLY Export with Outlier Removal

    static func saveAccumulatedPLY(points: [simd_float3], colors: [simd_uchar3]) -> URL? {
        guard !points.isEmpty else { return nil }

        // Remove statistical outliers before writing
        let (cleanPoints, cleanColors) = removeOutliers(points: points, colors: colors)
        guard !cleanPoints.isEmpty else { return nil }

        let fileName = "scan_\(Date().timeIntervalSince1970).ply"
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        guard let fileHandle = try? FileHandle(forWritingTo: fileURL) else { return nil }
        defer { fileHandle.closeFile() }

        func writeLine(_ s: String) {
            guard let data = (s + "\n").data(using: .ascii) else { return }
            fileHandle.write(data)
        }

        writeLine("ply")
        writeLine("format ascii 1.0")
        writeLine("element vertex \(cleanPoints.count)")
        writeLine("property float x")
        writeLine("property float y")
        writeLine("property float z")
        writeLine("property uchar red")
        writeLine("property uchar green")
        writeLine("property uchar blue")
        writeLine("end_header")
        for i in 0..<cleanPoints.count {
            let p = cleanPoints[i]
            let c = cleanColors[i]
            writeLine("\(p.x * 1000) \(p.y * 1000) \(p.z * 1000) \(c.x) \(c.y) \(c.z)")
        }
        return fileURL
    }

    // MARK: - Statistical Outlier Removal

    /// Removes isolated points that have fewer than `minNeighbors` occupied voxel neighbours
    /// within a 26-connected neighbourhood at `cellSize` resolution.
    private static func removeOutliers(
        points: [simd_float3],
        colors: [simd_uchar3],
        cellSize: Float = 0.002,
        minNeighbors: Int = 3
    ) -> ([simd_float3], [simd_uchar3]) {

        // Build occupancy set
        struct NeighborKey: Hashable { let xi, yi, zi: Int32 }
        var occupied = Set<NeighborKey>(minimumCapacity: points.count)
        for p in points {
            occupied.insert(NeighborKey(
                xi: Int32(floor(p.x / cellSize)),
                yi: Int32(floor(p.y / cellSize)),
                zi: Int32(floor(p.z / cellSize))
            ))
        }

        // Keep only points with enough occupied neighbours in the 26-connected neighbourhood
        let offsets: [Int32] = [-1, 0, 1]
        var outPoints = [simd_float3]()
        var outColors = [simd_uchar3]()
        outPoints.reserveCapacity(points.count)
        outColors.reserveCapacity(points.count)

        for i in 0..<points.count {
            let p = points[i]
            let bxi = Int32(floor(p.x / cellSize))
            let byi = Int32(floor(p.y / cellSize))
            let bzi = Int32(floor(p.z / cellSize))

            var count = 0
            outer: for dx in offsets {
                for dy in offsets {
                    for dz in offsets {
                        if dx == 0 && dy == 0 && dz == 0 { continue }
                        if occupied.contains(NeighborKey(xi: bxi + dx, yi: byi + dy, zi: bzi + dz)) {
                            count += 1
                            if count >= minNeighbors { break outer }
                        }
                    }
                }
            }

            if count >= minNeighbors {
                outPoints.append(p)
                outColors.append(colors[i])
            }
        }

        return (outPoints, outColors)
    }

    // MARK: - Pixel Buffer Utilities

    static func copyPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        var copy: CVPixelBuffer?
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferMetalCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixelFormat, attrs, &copy)
        guard status == kCVReturnSuccess, let outputBuffer = copy else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(outputBuffer, [])
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(outputBuffer, [])
        }
        if let src = CVPixelBufferGetBaseAddress(pixelBuffer),
           let dst = CVPixelBufferGetBaseAddress(outputBuffer) {
            let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let dstBytesPerRow = CVPixelBufferGetBytesPerRow(outputBuffer)
            let copyBytesPerRow = min(srcBytesPerRow, dstBytesPerRow)
            for row in 0..<height {
                memcpy(dst.advanced(by: row * dstBytesPerRow),
                       src.advanced(by: row * srcBytesPerRow),
                       copyBytesPerRow)
            }
        }
        return outputBuffer
    }

    static func convertYCbCrToBGRA(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        if pixelFormat == kCVPixelFormatType_32BGRA {
            return pixelBuffer
        }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var bgraPixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferMetalCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ] as CFDictionary
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &bgraPixelBuffer)
        guard status == kCVReturnSuccess, let outputBuffer = bgraPixelBuffer else { return nil }
        context.render(ciImage, to: outputBuffer)
        return outputBuffer
    }
}
