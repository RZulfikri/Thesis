/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The main view of the application.
*/

import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @ObservedObject private var processingQueue = ProcessingQueue.shared
    @AppStorage("topLabelText") private var topLabelText: String = "3d-label"
    @State private var showingHistory = false
    @State private var showOnlineNotSupported = false
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Camera View (2D Only)
            PreviewMetalViewRepresentable(cameraManager: cameraManager)
                .rotation3DEffect(.degrees(180), axis: (x: 0, y: 1, z: 0))
                .edgesIgnoringSafeArea(.all)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Top bar: history (left) | label (center)
            VStack {
                ZStack {
                    // Scan label — absolutely centered
                    Text(topLabelText)
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(10)
                        .onTapGesture {
                            presentLabelEditor()
                        }

                    // Leading / trailing controls
                    HStack(alignment: .center) {
                        // History button — top left (badge jika ada proses aktif)
                        Button(action: {
                            showingHistory = true
                        }) {
                            ZStack(alignment: .topTrailing) {
                                Image(systemName: "clock.arrow.circlepath")
                                    .font(.system(size: 20, weight: .medium))
                                    .foregroundColor(.white)
                                    .frame(width: 40, height: 40)
                                    .background(Color.black.opacity(0.6))
                                    .clipShape(Circle())

                                // Badge: jumlah task aktif
                                if processingQueue.activeCount > 0 {
                                    Text("\(processingQueue.activeCount)")
                                        .font(.system(size: 10, weight: .bold))
                                        .foregroundColor(.white)
                                        .frame(width: 17, height: 17)
                                        .background(Color.blue)
                                        .clipShape(Circle())
                                        .offset(x: 3, y: -3)
                                        .transition(.scale.combined(with: .opacity))
                                }
                            }
                            .animation(.spring(response: 0.3), value: processingQueue.activeCount)
                        }

                        Spacer()
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 16)

                Spacer()

                // Bottom: depth status + scan button
                VStack(spacing: 16) {
                    if !cameraManager.isScanning {
                        DepthStatusBadge(depthState: cameraManager.depthState)
                    } else {
                        DepthStatusBadge(depthState: cameraManager.depthState, override: "Scanning...")
                    }

                    if !cameraManager.isScanning {
                        Button(action: { cameraManager.startCountdown() }) {
                            ZStack {
                                Circle()
                                    .stroke(Color.white, lineWidth: 4)
                                    .frame(width: 80, height: 80)
                                Circle()
                                    .fill(cameraManager.depthState == .inRange ? Color.red : Color.gray)
                                    .frame(width: 70, height: 70)
                            }
                        }
                        .disabled(cameraManager.depthState != .inRange)
                    } else {
                        // Placeholder agar layout tidak bergeser saat scanning
                        Circle()
                            .stroke(Color.white.opacity(0.3), lineWidth: 4)
                            .frame(width: 80, height: 80)
                    }
                }
                .padding(.bottom, 30)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

            // UI Overlay (touchDepth / thermal / permission)
            VStack {
                Spacer()

                if !cameraManager.touchDepth.isEmpty {
                    Text(cameraManager.touchDepth)
                        .font(.largeTitle)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.black.opacity(0.5))
                        .cornerRadius(10)
                }

                if cameraManager.thermalState != .nominal {
                    Text("Thermal State: \(thermalStateString(cameraManager.thermalState))")
                        .foregroundColor(.red)
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(8)
                }

                if !cameraManager.permissionGranted {
                    Text("Camera Access Denied")
                        .foregroundColor(.red)
                        .padding()
                }

                Spacer()
                Spacer()
            }
        }
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .onReceive(cameraManager.$shouldShowHistory) { shouldShare in
            if shouldShare {
                showingHistory = true
                cameraManager.shouldShowHistory = false
            }
        }
        .onAppear {
            cameraManager.startSession()
            cameraManager.isJetEnabled = true
        }
        .onDisappear {
            cameraManager.stopSession()
        }
        .alert(
            isPresented: Binding(
                get: { cameraManager.alertMessage != nil },
                set: { if !$0 { cameraManager.alertMessage = nil } }
            )
        ) {
            let current = cameraManager.alertMessage ?? CameraManager.AlertMessage(title: "Error", message: "Unknown error")
            return Alert(
                title: Text(current.title),
                message: Text(current.message),
                dismissButton: .default(Text("OK"), action: {
                    cameraManager.alertMessage = nil
                })
            )
        }
        // .alert("Not Supported", isPresented: $showOnlineNotSupported) {
        //     Button("OK", role: .cancel) {}
        // } message: {
        //     Text("Cloud upload is not supported yet. Using local mode.")
        // }
        .sheet(isPresented: $showingHistory) {
            if #available(iOS 16.0, *) {
                ScanHistoryView()
                    .presentationDetents([.medium, .large])
                    .presentationDragIndicator(.visible)
            } else {
                ScanHistoryView()
            }
        }
    }
    
    func thermalStateString(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }
    
    private func presentLabelEditor() {
        let alert = UIAlertController(title: "Edit Label", message: nil, preferredStyle: .alert)
        alert.addTextField { textField in
            textField.placeholder = "3d-label"
            textField.text = topLabelText
            textField.clearButtonMode = .whileEditing
            textField.autocapitalizationType = .none
            textField.autocorrectionType = .no
        }
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        alert.addAction(UIAlertAction(title: "Save", style: .default, handler: { _ in
            let input = alert.textFields?.first?.text ?? ""
            let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
            topLabelText = trimmed.isEmpty ? "3d-label" : trimmed
        }))
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootViewController = windowScene.windows.first?.rootViewController {
            var topController = rootViewController
            while let presented = topController.presentedViewController {
                topController = presented
            }
            topController.present(alert, animated: true, completion: nil)
        }
    }
    
    /// Zips one or more scan folders into a single .zip file.
    /// Each folder is copied as-is into a staging directory (not pre-zipped),
    /// so the final archive contains plain folders — not nested .zip files.
    ///
    /// Structure of the resulting archive:
    ///   zipName.zip
    ///   ├── rahmat_20260505_134500/
    ///   │   ├── depth_0000.png
    ///   │   └── …
    ///   └── alji_20260505_143000/
    ///       └── …
    static func zipFolders(_ folderURLs: [URL], zipName: String) -> URL? {
        guard !folderURLs.isEmpty else { return nil }

        let fm  = FileManager.default
        let tmp = fm.temporaryDirectory

        // Staging dir: each scan folder is copied here, then the whole dir is zipped once.
        let stagingURL = tmp.appendingPathComponent(zipName + "_staging")
        try? fm.removeItem(at: stagingURL)
        do {
            try fm.createDirectory(at: stagingURL, withIntermediateDirectories: true)
        } catch {
            print("zipFolders: staging dir creation failed: \(error)"); return nil
        }

        // Copy each scan folder directly (folder stays as folder, not pre-zipped).
        for folderURL in folderURLs {
            let dest = stagingURL.appendingPathComponent(folderURL.lastPathComponent)
            do {
                try fm.copyItem(at: folderURL, to: dest)
            } catch {
                print("zipFolders: copy failed for \(folderURL.lastPathComponent): \(error)")
            }
        }

        guard (try? fm.contentsOfDirectory(at: stagingURL, includingPropertiesForKeys: nil))?.isEmpty == false else {
            print("zipFolders: staging dir empty, all copies failed"); return nil
        }

        // Zip the staging dir once into the final archive.
        let finalZipURL = tmp.appendingPathComponent(zipName + ".zip")
        try? fm.removeItem(at: finalZipURL)

        var coordError: NSError?
        NSFileCoordinator().coordinate(readingItemAt: stagingURL,
                                        options: .forUploading,
                                        error: &coordError) { tempZipURL in
            do {
                try fm.copyItem(at: tempZipURL, to: finalZipURL)
            } catch let copyError {
                print("zipFolders: final copy failed: \(copyError)")
            }
        }
        try? fm.removeItem(at: stagingURL)

        if let coordError { print("zipFolders: coordinator error: \(coordError)"); return nil }
        guard fm.fileExists(atPath: finalZipURL.path) else {
            print("zipFolders: final zip not found at \(finalZipURL.path)"); return nil
        }
        return finalZipURL
    }
}

// MARK: - Palm Status Badge

struct DepthStatusBadge: View {
    let depthState: CameraManager.DepthState
    var override: String? = nil

    private var label: String {
        if let text = override { return text }
        switch depthState {
        case .noObject:  return "Position hand in front of camera"
        case .tooClose:  return "Too close — move hand away"
        case .tooFar:    return "Too far — bring hand closer"
        case .inRange:   return "Ready to scan"
        }
    }

    private var color: Color {
        if override != nil { return .blue }
        switch depthState {
        case .noObject:  return .orange
        case .tooClose:  return .red
        case .tooFar:    return .red
        case .inRange:   return .green
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
        }
        .font(.subheadline.weight(.semibold))
        .foregroundColor(color)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.black.opacity(0.6))
        .cornerRadius(20)
        .animation(.easeInOut(duration: 0.2), value: depthState)
    }
}

// MARK: - Palm Depth Mask Overlay

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
