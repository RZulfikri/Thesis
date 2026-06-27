/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
SwiftUI wrapper for PreviewMetalView.
*/

import SwiftUI
import AVFoundation

struct PreviewMetalViewRepresentable: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager

    func makeUIView(context: Context) -> PreviewMetalView {
        let view = PreviewMetalView(frame: .zero, device: nil)

        // Ensure view fills container
        view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.contentMode = .scaleAspectFill

        // Add gestures
        let tapGesture = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap(_:)))
        view.addGestureRecognizer(tapGesture)

        let pressGesture = UILongPressGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleLongPress(_:)))
        pressGesture.minimumPressDuration = 0.05
        pressGesture.cancelsTouchesInView = false
        view.addGestureRecognizer(pressGesture)

        // Provide the resolver to CameraManager
        // We need to capture the view weakly to avoid retain cycles
        cameraManager.textureTransformResolver = { [weak view] point in
            return view?.texturePointForView(point: point)
        }

        return view
    }

    func updateUIView(_ uiView: PreviewMetalView, context: Context) {
        uiView.pixelBuffer = cameraManager.previewPixelBuffer

        // Derive rotation from current interface orientation.
        // TrueDepth front camera delivers buffers in landscapeLeft orientation.
        let scene = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first(where: { $0.activationState == .foregroundActive })
        let interfaceOrientation = scene?.interfaceOrientation ?? .portrait

        if let rotation = PreviewMetalView.Rotation(
            with: interfaceOrientation,
            videoOrientation: .landscapeLeft,
            cameraPosition: .front
        ) {
            uiView.rotation = rotation
        } else {
            uiView.rotation = .rotate90Degrees
        }
        uiView.mirroring = false
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject {
        var parent: PreviewMetalViewRepresentable

        init(_ parent: PreviewMetalViewRepresentable) {
            self.parent = parent
        }

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let view = gesture.view as? PreviewMetalView else { return }
            let location = gesture.location(in: view)
            if let normalizedPoint = view.normalizedTexturePointForView(point: location) {
                parent.cameraManager.focusAndExpose(at: normalizedPoint)
            }
        }

        @objc func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
            guard let view = gesture.view as? PreviewMetalView else { return }

            switch gesture.state {
            case .began:
                let point = gesture.location(in: view)
                parent.cameraManager.setTouchCoordinates(point, detected: true)
            case .changed:
                let point = gesture.location(in: view)
                parent.cameraManager.setTouchCoordinates(point, detected: true)
            case .ended, .cancelled, .failed:
                parent.cameraManager.setTouchCoordinates(.zero, detected: false)
            default:
                break
            }
        }
    }
}
