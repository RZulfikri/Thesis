# Object Scan Feature Implementation Plan (PLY Output)

I will implement the object scanning feature that captures the TrueDepth camera output as a `.ply` point cloud file. The workflow includes a countdown, a scanning phase, and a result preview.

## 1. Data Model & PLY Export (`CameraManager.swift`)
- [ ] **State Management**:
    - Add `isScanning` (Bool) to track the active state.
    - Add `isCountingDown` (Bool) and `countdownValue` (Int).
    - Add `capturedDepthData` (`AVDepthData?`) and `capturedVideoTexture` (`CVPixelBuffer?`) to hold the result for preview.
    - Add `plyFileURL` (`URL?`) to store the path of the saved file.
- [ ] **Scanning Logic**:
    - **Start**: `startCountdown()` -> decrements 3, 2, 1 -> sets `isScanning = true`.
    - **Stop**: `stopScanning()` -> captures the *current* frame's depth and video -> triggers PLY generation -> sets `isScanning = false`.
    - **Note**: "Scanning" in this context will be interpreted as "Previewing until capture". Since we are outputting a *single* PLY file (as typical for this kind of "scan object" request without advanced reconstruction), we will capture the frame at the moment of "Stop" (or accumulate if feasible, but single frame is safer and robust). To support "highest heatmap object" (closest object), we can optionally buffer the frame with the nearest mean depth during the scan period, but a simple "capture on stop" is the standard interpretation for manual control. I will implement "capture on stop" for simplicity and reliability.
- [ ] **PLY Writer**:
    - Implement `saveToPLY(depthData: AVDepthData, texture: CVPixelBuffer) -> URL`.
    - This function will:
        - Access the depth buffer (Float16/32).
        - Access the color buffer (BGRA).
        - Use the intrinsic matrix (`depthData.cameraCalibrationData.intrinsicMatrix`) to unproject pixels (u, v, z) to 3D points (x, y, z).
        - Write the header and vertex list to a text-based `.ply` file in the temporary directory.

## 2. UI Implementation (`ContentView.swift` & Subviews)
- [ ] **Create `ScannerControlsView.swift`**:
    - **Start Button**: Visible when idle. Starts countdown.
    - **Countdown Overlay**: Big text "3", "2", "1" over the live 2D view.
    - **Stop Button**: Visible when `isScanning`. Triggers capture.
- [ ] **Create `ResultPreviewView.swift`**:
    - Displays the captured point cloud.
    - **Reuse `PointCloudMetalView`**: Since the Metal view expects `AVDepthData`, we can simply pass the `capturedDepthData` and `capturedVideoTexture` to it.
    - **Restore `PointCloudViewRepresentable.swift`**: I will recreate this file (which I deleted) to wrap `PointCloudMetalView` for SwiftUI.
    - **UI**: A "Save/Done" button to dismiss the preview and return to the live camera.
- [ ] **Update `ContentView.swift`**:
    - Show `ScannerControlsView` over the 2D preview.
    - Present `ResultPreviewView` (sheet or full screen) when a capture is available.

## 3. Project Configuration
- [ ] **Update `TrueDepthStreamer.xcodeproj`**:
    - Add `PointCloudViewRepresentable.swift` (newly created) back to the project.
    - Ensure `ScannerControlsView.swift` and `ResultPreviewView.swift` (if separate files) are added. (I will likely put them in `ContentView.swift` or a new `Views.swift` file to minimize project file churn).

## 4. Verification
- [ ] **Build**: Ensure all files are linked.
- [ ] **Flow**: Start -> 3..2..1 -> Scan (Live 2D) -> Stop -> Preview (Static 3D Point Cloud).
- [ ] **Output**: Verify the `.ply` file is generated (I can print the path to console).
