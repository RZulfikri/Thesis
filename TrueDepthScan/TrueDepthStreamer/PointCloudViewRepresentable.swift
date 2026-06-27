/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
SwiftUI wrapper for PointCloudMetalView.
*/

import SwiftUI
import AVFoundation

struct PointCloudViewRepresentable: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager
    var autoPanningEnabled: Bool = true
    
    // Allow passing static data for preview
    var staticDepthData: AVDepthData?
    var staticVideoTexture: CVPixelBuffer?
    
    func makeUIView(context: Context) -> PointCloudMetalView {
        let view = PointCloudMetalView(frame: .zero, device: nil)
        
        // Add gestures
        let pinchGesture = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
        view.addGestureRecognizer(pinchGesture)
        
        let panGesture = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePan(_:)))
        panGesture.maximumNumberOfTouches = 1
        panGesture.minimumNumberOfTouches = 1
        view.addGestureRecognizer(panGesture)
        
        let doubleTapGesture = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleDoubleTap(_:)))
        doubleTapGesture.numberOfTapsRequired = 2
        doubleTapGesture.numberOfTouchesRequired = 1
        view.addGestureRecognizer(doubleTapGesture)
        
        let rotateGesture = UIRotationGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleRotate(_:)))
        view.addGestureRecognizer(rotateGesture)
        
        return view
    }
    
    func updateUIView(_ uiView: PointCloudMetalView, context: Context) {
        // Auto panning logic (only if live)
        if staticDepthData == nil && autoPanningEnabled {
             context.coordinator.performAutoPan(view: uiView)
        } else {
             context.coordinator.resetAutoPan()
        }
        
        if let depthData = staticDepthData, let videoTexture = staticVideoTexture {
            uiView.setDepthFrame(depthData, withTexture: videoTexture)
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject {
        var parent: PointCloudViewRepresentable
        
        // Gesture State
        private var lastScale = Float(1.0)
        private var lastZoom = Float(0.0)
        private var lastXY = CGPoint(x: 0, y: 0)
        
        // Auto Pan State
        private var autoPanningIndex = -1
        
        init(_ parent: PointCloudViewRepresentable) {
            self.parent = parent
        }
        
        func resetAutoPan() {
            autoPanningIndex = -1
        }
        
        func performAutoPan(view: PointCloudMetalView) {
            if autoPanningIndex == -1 {
                autoPanningIndex = 0
            }
            
            let moves = 200
            let factor = 2.0 * .pi / Double(moves)
            
            let pitch = sin(Double(autoPanningIndex) * factor) * 2
            let yaw = cos(Double(autoPanningIndex) * factor) * 2
            autoPanningIndex = (autoPanningIndex + 1) % moves
            
            view.resetView()
            view.pitchAroundCenter(Float(pitch) * 10)
            view.yawAroundCenter(Float(yaw) * 10)
        }
        
        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let view = gesture.view as? PointCloudMetalView else { return }
            
            if gesture.numberOfTouches != 2 { return }
            
            if gesture.state == .began {
                lastScale = 1
            } else if gesture.state == .changed {
                let scale = Float(gesture.scale)
                let diff: Float = scale - lastScale
                let factor: Float = 1e3
                if scale < lastScale {
                    lastZoom = diff * factor
                } else {
                    lastZoom = diff * factor
                }
                
                view.moveTowardCenter(lastZoom)
                lastScale = scale
            }
        }
        
        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let view = gesture.view as? PointCloudMetalView else { return }
            
            if gesture.numberOfTouches != 1 { return }
            
            if gesture.state == .began {
                lastXY = gesture.translation(in: view)
            } else if gesture.state != .failed && gesture.state != .cancelled {
                let pnt = gesture.translation(in: view)
                
                view.yawAroundCenter(Float((pnt.x - lastXY.x) * 0.1))
                view.pitchAroundCenter(Float((pnt.y - lastXY.y) * 0.1))
                lastXY = pnt
            }
        }
        
        @objc func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
            guard let view = gesture.view as? PointCloudMetalView else { return }
            view.resetView()
        }
        
        @objc func handleRotate(_ gesture: UIRotationGestureRecognizer) {
            guard let view = gesture.view as? PointCloudMetalView else { return }
            
            if gesture.numberOfTouches != 2 { return }
            
            if gesture.state == .changed {
                let rot = Float(gesture.rotation)
                view.rollAroundCenter(rot * 60)
                gesture.rotation = 0
            }
        }
    }
}
