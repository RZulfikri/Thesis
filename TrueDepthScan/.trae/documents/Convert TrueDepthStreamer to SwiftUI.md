# Migration Plan: UIKit to SwiftUI

I will convert the `TrueDepthStreamer` project from UIKit/Storyboards to SwiftUI. This involves separating the camera logic from the view controller, creating SwiftUI wrappers for the Metal views, and rebuilding the UI.

## 1. Project Configuration
- [ ] **Create `TrueDepthStreamerApp.swift`**: Define the SwiftUI `App` entry point.
- [ ] **Update `Info.plist`**: Remove the `UIMainStoryboardFile` key to stop launching the storyboard.
- [ ] **Delete Legacy Files**: Remove `AppDelegate.swift`, `CameraViewController.swift`, and `Main.storyboard`.

## 2. Core Logic Extraction (`CameraManager`)
- [ ] **Create `CameraManager.swift`**: An `ObservableObject` that manages the `AVCaptureSession`.
    - **Responsibilities**:
        - Session configuration and permission handling.
        - `AVCaptureDataOutputSynchronizerDelegate` implementation.
        - Video and Depth mixing logic (using `VideoMixer` and `DepthToJETConverter`).
        - Thermal state monitoring.
    - **Published Properties**:
        - `previewPixelBuffer`: For the 2D JET view.
        - `depthData` & `videoTexture`: For the 3D Point Cloud view.
        - `isJetEnabled`, `isSmoothingEnabled`, `mixFactor`: UI state.
        - `touchDepth`: Text for the depth measurement label.
        - `alertMessage`: For showing permission/error alerts.

## 3. Metal View Wrappers
- [ ] **Create `PreviewMetalViewRepresentable.swift`**:
    - Wraps `PreviewMetalView` using `UIViewRepresentable`.
    - Binds `pixelBuffer`, `rotation`, and `mirroring` from `CameraManager`.
    - **Gestures**: Implement a `Coordinator` to handle Tap (Focus) and Long Press (Depth Measurement) gestures, forwarding actions to `CameraManager`.
- [ ] **Create `PointCloudViewRepresentable.swift`**:
    - Wraps `PointCloudMetalView` using `UIViewRepresentable`.
    - Updates the view with new `depthData` and `videoTexture` in `updateUIView`.
    - **Gestures**: Implement a `Coordinator` to attach and handle `UIPinchGestureRecognizer` (Zoom), `UIPanGestureRecognizer` (Rotate), `UITapGestureRecognizer` (Reset), and `UIRotationGestureRecognizer` (Roll), replicating the logic from `CameraViewController`.

## 4. UI Implementation
- [ ] **Create `ControlView.swift`**:
    - A SwiftUI view containing the controls:
        - Segmented Control (2D / 3D).
        - Smoothing Switch.
        - Mix Factor Slider.
        - Auto Panning Switch.
        - Status Labels (Camera Unavailable, Thermal State, Depth Value).
- [ ] **Create `ContentView.swift`**:
    - The main screen composing the UI.
    - Uses `ZStack` to layer the Metal views (switched based on `isJetEnabled`) and the `ControlView`.
    - Connects `CameraManager` to the views.

## 5. Verification
- [ ] **Compile & Run**: Ensure the app builds without errors.
- [ ] **Test Functionality**:
    - Camera permissions and startup.
    - Switching between 2D (JET) and 3D (Point Cloud) modes.
    - Depth smoothing and mix factor adjustments.
    - Point Cloud gestures (Zoom, Rotate, Reset).
    - Depth measurement on touch.
