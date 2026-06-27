# Object Scan Feature Implementation Plan

I will implement the "Scan Object" feature using the TrueDepth camera. This involves adding a recording mechanism, a countdown timer, and a preview screen for the captured result. The "highest heatmap object" will be interpreted as recording the processed JET (heatmap) video stream.

## 1. Data Model & Logic (`CameraManager.swift`)
- [ ] **State Management**:
    - Add `isRecording` (Bool) to track recording state.
    - Add `isCountingDown` (Bool) and `countdownValue` (Int) for the timer.
    - Add `recordedVideoURL` (URL?) to store the path of the captured video.
- [ ] **Video Recording Logic**:
    - Implement `AVAssetWriter` setup to record the `previewPixelBuffer` (which contains the heatmap mixed with video).
    - Create `startRecording()` method:
        - Sets up `AVAssetWriter`.
        - Starts writing frames from `dataOutputSynchronizer` when `isRecording` is true.
    - Create `stopRecording()` method:
        - Finishes writing.
        - Updates `recordedVideoURL`.
        - Sets `isRecording` to false.
    - Update `dataOutputSynchronizer`:
        - Append `previewPixelBuffer` to the asset writer if recording.
- [ ] **Countdown Logic**:
    - Implement a `startCountdown()` method that decrements `countdownValue` from 3 to 0 using a timer/dispatch, then calls `startRecording()`.

## 2. UI Implementation
- [ ] **Create `ScanControlsView.swift`**:
    - A subview for the bottom controls.
    - **Start Button**:
        - Visible when not recording.
        - Triggers `cameraManager.startCountdown()`.
    - **Stop Button**:
        - Visible when recording (`isRecording == true`).
        - Triggers `cameraManager.stopRecording()`.
    - **Countdown Overlay**:
        - Displays large text (3, 2, 1) over the center of the screen when `isCountingDown` is true.
- [ ] **Create `ResultPreviewView.swift`**:
    - A view to play back the recorded video.
    - Uses `AVPlayer` and `VideoPlayer` (from `AVKit`) to show the result.
    - Includes a "Done" or "Close" button to dismiss and return to the camera.
- [ ] **Update `ContentView.swift`**:
    - Integrate `ScanControlsView`.
    - Present `ResultPreviewView` as a sheet or full-screen cover when `recordedVideoURL` is populated.

## 3. Asset Writer Helper (`MovieRecorder.swift`)
- [ ] **Create `MovieRecorder.swift`**:
    - A helper class to encapsulate `AVAssetWriter` logic (setup, append sample buffer, finish writing).
    - Handles audio (optional, but good practice) and video inputs.
    - Manages file system paths for temporary video storage.

## 4. Verification
- [ ] **Compile & Run**: Ensure no build errors.
- [ ] **Test Countdown**: Verify 3-2-1 count before recording starts.
- [ ] **Test Recording**: Verify that the JET heatmap video is actually written to a file.
- [ ] **Test Preview**: Verify the recorded video plays back correctly in the result screen.
- [ ] **Test Cycle**: Start -> Countdown -> Record -> Stop -> Preview -> Close -> Start again.
