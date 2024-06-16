//
//  ContentView.swift
//  CreateMLSwiftUISample
//
//  Created by 永井涼 on 2024/06/16.
//

import SwiftUI
import AVFoundation
import Vision
import CoreML

struct CameraView: UIViewControllerRepresentable {
    class Coordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
        var parent: CameraView
        var imageClassifierVisionModel: VNCoreMLModel
        
        init(parent: CameraView) {
            self.parent = parent
            
            let defaultConfig = MLModelConfiguration()
            let imageClassifierWrapper = try? MyPetCatClassifier(configuration: defaultConfig)
            guard let imageClassifier = imageClassifierWrapper else {
                fatalError("App failed to create an image classifier model instance.")
            }
            let imageClassifierModel = imageClassifier.model
            self.imageClassifierVisionModel = try! VNCoreMLModel(for: imageClassifierModel)
        }
        
        func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let request = VNCoreMLRequest(model: imageClassifierVisionModel) { (request, error) in
                if let results = request.results as? [VNClassificationObservation] {
                    if let firstResult = results.first {
                        let confidence = firstResult.confidence
                        let identifier = firstResult.identifier
                        DispatchQueue.main.async {
                            if confidence < 0.3 {
                                self.parent.classificationLabel = "当てはまらない (信頼度: \(Int(confidence * 100))%)"
                            } else {
                                self.parent.classificationLabel = "\(identifier) (信頼度: \(Int(confidence * 100))%)"
                            }
                        }
                    }
                }
            }
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            try? handler.perform([request])
        }
    }
    
    @Binding var classificationLabel: String
    
    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }
    
    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = UIViewController()
        let captureSession = AVCaptureSession()
        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else { return viewController }
        let videoInput = try? AVCaptureDeviceInput(device: videoCaptureDevice)
        if (captureSession.canAddInput(videoInput!)) {
            captureSession.addInput(videoInput!)
        }
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(context.coordinator, queue: DispatchQueue(label: "videoQueue"))
        if (captureSession.canAddOutput(videoOutput)) {
            captureSession.addOutput(videoOutput)
        }
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = viewController.view.layer.bounds
        previewLayer.videoGravity = .resizeAspectFill
        viewController.view.layer.addSublayer(previewLayer)
        
        captureSession.startRunning()
        
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
    }
}

struct ContentView: View {
    @State private var classificationLabel: String = "Initializing..."
    
    var body: some View {
        VStack {
            CameraView(classificationLabel: $classificationLabel)
                .edgesIgnoringSafeArea(.all)
            Text(classificationLabel)
                .padding()
                .background(Color.white)
                .cornerRadius(10)
                .shadow(radius: 10)
        }
    }
}

// 3. プレビュー
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
