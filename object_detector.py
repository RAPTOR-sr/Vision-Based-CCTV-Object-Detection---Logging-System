# live_object_captioner/object_detector.py
from ultralytics import YOLO
import cv2

class ObjectDetector:
    """
    Handles object detection using a YOLO model.
    """
    def __init__(self, model_path='yolov26n.pt'):
        """
        Initializes the detector.

        Args:
            model_path (str): Path to the YOLO model file (e.g., 'yolov8n.pt').
                              'yolovXx.pt' is small and fast.
        """
        try:
            self.model = YOLO(model_path)
            # Determine the best device to use
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = 'cuda'
                self.model.to('cuda')
                print(f"YOLO model '{model_path}' loaded successfully on GPU (CUDA).")
            else:
                self.device = 'cpu'
                self.model.to('cpu')
                print(f"YOLO model '{model_path}' loaded successfully on CPU (CUDA not available).")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Please ensure the model file exists and dependencies are installed.")
            self.model = None # Indicate model loading failure

    def detect_objects(self, frame):
        if self.model is None:
            print("Error: YOLO model not loaded")
            return []

        try:
            # Add frame shape check
            if frame is None:
                print("Error: Empty frame received")
                return []
                
            print(f"Frame shape: {frame.shape}")
            print("Running YOLO inference...")
            
            # Use the device determined during initialization
            results = self.model(frame, device=self.device, imgsz=320)
            
            detections = []
            for result in results:
                boxes = result.boxes
                print(f"Found {len(boxes)} objects")
                
                for box in boxes:
                    # Convert tensors to CPU before numpy conversion
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    print(f"Detected {class_name} at ({x1},{y1},{x2},{y2}) conf={confidence:.2f}")
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_name': class_name
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_model_names(self):
        """Returns the class names the model can detect."""
        if self.model:
            return self.model.names
        return {}