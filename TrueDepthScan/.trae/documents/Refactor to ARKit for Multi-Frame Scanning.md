# Implement Multi-Frame Point Cloud Fusion (Accumulation)

The user wants to accumulate point clouds as the camera moves slowly, ensuring a smooth merged result, and stop if tracking is lost (frames differ too much). This effectively changes the app from a "snapshot" scanner to a "real-time fusion" scanner (like a simplified SLAM/KinectFusion).

Implementing full ICP/SLAM in Swift from scratch is complex and slow on CPU. However, for a single object (hand), we can use a simpler approach or leverage Metal. But given the constraints and the current codebase, I will implement a **simplified accumulation strategy** based on camera pose tracking (if available) or basic registration.

**Wait**, the user asked: "ensure that when the camera moves slowly, point clouds merge... and stop if...".
Standard `AVDepthData` does **not** provide camera pose relative to the world (trajectory) by itself. We need **ARKit** (`ARSession`) to track the camera movement (`ARCamera.transform`).
Without ARKit, we cannot align frames unless we do heavy ICP (Iterative Closest Point) which is hard to do in real-time in Swift without external libraries.

**Pivot**: I must switch the backend from `AVCaptureSession` to **`ARSession`** (ARKit).
ARKit provides:
1.  **Depth Data** (`ARFrame.sceneDepth` or `capturedDepthData`).
2.  **Camera Pose** (`ARCamera.transform`): Essential for placing points in a shared world coordinate system.
3.  **Tracking State**: Tells us if tracking is valid or lost.

## Plan

### 1. Switch to ARKit (`CameraManager.swift`)
- [ ] Replace `AVCaptureSession` with `ARSession`.
- [ ] Configure `ARWorldTrackingConfiguration` with `frameSemantics = .sceneDepth` (if available on device) or standard `.capturedDepthData`.
- [ ] Implement `ARSessionDelegate`.

### 2. Implement Accumulation Logic
- [ ] **World Space Transformation**: For each frame, transform points from **Camera Space** to **World Space** using `frame.camera.transform`.
- [ ] **Voxel Grid / Point Fusion**:
    -   Simply appending points will create massive files and noise.
    -   I will implement a simple **Global Point Buffer**.
    -   **Filtering**: Only add points that are "new" or merge them?
    -   *Simpler Approach for this session*: Just append world-space points from valid frames (every Nth frame or when movement > threshold).
    -   **Validation**: Check `ARCamera.trackingState`. If `limited` or `notAvailable`, stop/pause scanning.

### 3. Stop Condition (Tracking Loss)
- [ ] Monitor `camera.trackingState`.
- [ ] Calculate camera movement speed. If too fast (blur risk), pause or stop.

### 4. Export
- [ ] The `saveToPLY` function will now export the **accumulated global points**, not just the last frame.

**Complexity Warning**: Switching to ARKit is a significant change.
*Alternative*: If the user *really* wants to keep `AVCaptureSession`, I'd have to implement ICP. That is extremely error-prone and slow. ARKit is the standard Apple way to do "scanning".

**Decision**: I will refactor `CameraManager` to use `ARKit` for scanning. This is the only robust way to "merge point clouds" while moving the camera.

## Step-by-Step Implementation

1.  **Refactor `CameraManager`**:
    -   Add `ARKit` import.
    -   Replace `AVCaptureSession` setup with `ARSession` setup.
    -   Delegate method `session(_:didUpdate:)` will handle data processing.
2.  **Accumulation**:
    -   Create a buffer `var accumulatedPoints: [Vector3]`.
    -   In delegate:
        -   Get `depthMap` and `camera.transform`.
        -   Unproject points to local space -> multiply by `camera.transform` -> World Space.
        -   Append to `accumulatedPoints`.
        -   (Optimization) Only keep points in the target range (10-35cm relative to camera).
3.  **Tracking Checks**:
    -   If `camera.trackingState != .normal`, stop accumulation.
4.  **Export**:
    -   Write `accumulatedPoints` to PLY.

This will allow the user to "paint" the object in 3D.

**Note**: I will keep the existing `AVCapture` logic if possible, or fully replace it. Full replacement is cleaner as they conflict for camera resource.

Let's do this.
