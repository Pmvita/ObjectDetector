#!/usr/bin/env python3
"""
Real-Time Object Detection using OpenCV and TensorFlow COCO-SSD model.
Detects objects from webcam feed and displays bounding boxes with labels.
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    HAS_TF_OBJECT_DETECTION = True
except ImportError:
    HAS_TF_OBJECT_DETECTION = False

# MobileNet-SSD class names (20 classes)
# Note: MobileNet-SSD uses different class IDs than full COCO dataset
MOBILENET_SSD_CLASSES = [
    'background',  # 0
    'aeroplane',   # 1
    'bicycle',     # 2
    'bird',        # 3
    'boat',        # 4
    'bottle',      # 5
    'bus',         # 6
    'car',         # 7
    'cat',         # 8
    'chair',       # 9
    'cow',         # 10
    'diningtable', # 11
    'dog',         # 12
    'horse',       # 13
    'motorbike',   # 14
    'person',      # 15
    'pottedplant', # 16
    'sheep',       # 17
    'sofa',        # 18
    'train',       # 19
    'tvmonitor'    # 20
]


class COCODetector:
    """COCO-SSD Object Detector using OpenCV DNN."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.load_model()
    
    def load_model(self):
        """Load COCO-SSD model using OpenCV DNN."""
        # Model files (we'll use a pre-trained MobileNet-SSD model)
        # These are standard COCO-SSD model files
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "MobileNetSSD_deploy.caffemodel"
        config_path = model_dir / "MobileNetSSD_deploy.prototxt"
        
        # Check if model files exist
        if not model_path.exists() or not config_path.exists():
            print("✗ Model files not found.")
            print("\nTo download the model files, run:")
            print("  python3 models/download_models.py")
            print("\nOr download manually from:")
            print("  Model: https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel")
            print("  Config: https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.prototxt")
            print("\nFor now, using a simplified detection approach...")
            self.net = None
            return
        
        try:
            print("Loading COCO-SSD model...")
            self.net = cv2.dnn.readNetFromCaffe(str(config_path), str(model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("✓ COCO-SSD model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Using fallback detection method...")
            self.net = None
    
    def detect(self, frame: np.ndarray) -> list:
        """Detect objects in frame."""
        if self.net is None:
            return []
        
        h, w = frame.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )
        
        # Set input and forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                class_id = int(detections[0, 0, i, 1])
                
                # Get bounding box coordinates
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Map to MobileNet-SSD classes
                if 0 <= class_id < len(MOBILENET_SSD_CLASSES):
                    class_name = MOBILENET_SSD_CLASSES[class_id]
                    results.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'class': class_name,
                        'confidence': confidence
                    })
        
        return results


class YOLODetector:
    """YOLO Object Detector - easier to set up, downloads automatically."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model (downloads automatically if needed)."""
        if not HAS_YOLO:
            print("YOLO not available. Install with: pip install ultralytics")
            return
        
        try:
            print("Loading YOLO model (this will download automatically if needed)...")
            # YOLOv8n is the nano version - fast and lightweight
            self.model = YOLO('yolov8n.pt')  # Downloads automatically
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> list:
        """Detect objects in frame."""
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence > self.confidence_threshold:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                            'class': class_name,
                            'confidence': confidence
                        })
            
            return detections
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            return []


class SimpleDetector:
    """Simple fallback detector using basic computer vision."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    def detect(self, frame: np.ndarray) -> list:
        """Simple motion/object detection using contours."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                results.append({
                    'bbox': (x, y, w, h),
                    'class': 'object',
                    'confidence': min(0.9, area / 10000)
                })
        
        return results


def draw_detections(frame: np.ndarray, detections: list):
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        x, y, w, h = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        # Skip background class
        if class_name == 'background':
            continue
        
        # Color based on confidence
        if confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif confidence > 0.5:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Draw bounding box (thicker for visibility)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Format label: Object name with confidence percentage
        # Make object name more readable
        label_text = class_name.replace('_', ' ').title()  # Convert "diningtable" to "Dining Table"
        confidence_text = f"{confidence * 100:.1f}%"
        
        # Calculate text sizes - make labels bigger and more visible
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Increased from 0.7
        thickness = 2
        
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        (conf_width, conf_height), _ = cv2.getTextSize(
            confidence_text, font, font_scale * 0.8, thickness
        )
        
        # Total label box dimensions
        total_width = max(label_width, conf_width) + 10
        total_height = label_height + conf_height + baseline + 15
        
        # Ensure label doesn't go off screen
        label_x = max(0, x)
        # Position label above bounding box, or below if too close to top
        if y < total_height + 10:
            label_y = y + h + total_height + 5  # Below box
        else:
            label_y = y  # Above box
        
        # Draw label background rectangle with border
        cv2.rectangle(
            frame,
            (label_x - 1, label_y - total_height - 1),
            (label_x + total_width + 1, label_y + 1),
            (0, 0, 0),  # Black border
            2
        )
        cv2.rectangle(
            frame,
            (label_x, label_y - total_height),
            (label_x + total_width, label_y),
            color,
            -1  # Filled rectangle
        )
        
        # Draw object name (larger, bold) - make it more prominent
        text_y = label_y - conf_height - baseline - 5 if y >= total_height + 10 else label_y - conf_height - baseline - 5
        cv2.putText(
            frame,
            label_text,
            (label_x + 5, text_y),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
        
        # Draw confidence percentage (smaller, below name)
        conf_y = label_y - 5 if y >= total_height + 10 else label_y - 5
        cv2.putText(
            frame,
            confidence_text,
            (label_x + 5, conf_y),
            font,
            font_scale * 0.8,
            (255, 255, 255),  # White text
            thickness
        )


def main():
    parser = argparse.ArgumentParser(description='Real-Time Object Detection')
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold (0.0-1.0)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Video width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Video height'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save video to file (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize detector - try YOLO first (easiest), then COCO-SSD, then fallback
    print("Initializing detector...")
    detector = None
    
    # Try YOLO first (easiest, downloads automatically)
    if HAS_YOLO:
        print("Trying YOLO detector (recommended)...")
        yolo_detector = YOLODetector(confidence_threshold=args.confidence)
        if yolo_detector.model is not None:
            detector = yolo_detector
            print("✓ Using YOLO model - object names will be displayed!")
        else:
            print("YOLO failed, trying COCO-SSD...")
    
    # Try COCO-SSD if YOLO didn't work
    if detector is None:
        coco_detector = COCODetector(confidence_threshold=args.confidence)
        if coco_detector.net is not None:
            detector = coco_detector
            print("✓ Using COCO-SSD model - object names will be displayed!")
        else:
            print("\n" + "="*60)
            print("⚠ WARNING: No ML models available!")
            print("="*60)
            print("The fallback detector will only show 'OBJECT' labels.")
            print("\nTo see actual object names, install YOLO (recommended):")
            print("  pip install ultralytics")
            print("\nOr download COCO-SSD model files:")
            print("  python3 models/download_models.py")
            print("="*60 + "\n")
            detector = SimpleDetector(confidence_threshold=args.confidence)
    
    # Initialize video capture
    print(f"Opening camera {args.camera}...")
    
    # On macOS, try to use AVFoundation backend (better permission handling)
    import platform
    if platform.system() == 'Darwin':
        # Try AVFoundation first (macOS native)
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            # Fallback to default backend
            cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"✗ Error: Could not open camera {args.camera}")
        print("\nTroubleshooting:")
        print("  1. Make sure your camera is connected and not being used by another app")
        print("  2. Try a different camera index: --camera 1 or --camera 2")
        if platform.system() == 'Darwin':
            print("  3. On macOS: Grant camera permission to Terminal/Python:")
            print("     - System Preferences > Security & Privacy > Camera")
            print("     - Check 'Terminal' or 'Python' in the list")
            print("     - Or run: python3 check_camera.py to diagnose")
        print("  4. Try restarting your terminal/IDE")
        sys.exit(1)
    
    # Set video properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Test camera by reading a frame (with retries for macOS)
    print("Testing camera...")
    ret, test_frame = None, None
    
    # Try multiple times (macOS sometimes needs a moment)
    for attempt in range(5):
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            break
        time.sleep(0.2)  # Wait 200ms between attempts
    
    if not ret or test_frame is None:
        print("✗ Error: Camera opened but could not read frames")
        print("\nThis is usually a macOS permission issue.")
        print("\nTo fix:")
        if platform.system() == 'Darwin':
            print("  1. Open System Preferences (or System Settings)")
            print("  2. Go to Security & Privacy > Privacy > Camera")
            print("  3. Find 'Terminal' (or 'iTerm', 'Python', etc.) in the list")
            print("  4. Check the box to enable camera access")
            print("  5. If Terminal is not listed:")
            print("     - Run: python3 check_camera.py")
            print("     - Or try: tccutil reset Camera")
            print("     - Then restart Terminal and try again")
        print("\nAlso try:")
        print("  - Close other apps using the camera (Zoom, FaceTime, etc.)")
        print("  - Run: python3 check_camera.py to diagnose")
        print("  - Try a different terminal app (iTerm2, VS Code terminal, etc.)")
        cap.release()
        sys.exit(1)
    
    print(f"✓ Camera working! Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Video writer (if saving)
    video_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.save,
            fourcc,
            20.0,
            (args.width, args.height)
        )
    
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save screenshot")
    print("  Press 'c' to change confidence threshold")
    print("\nStarting detection...\n")
    
    # FPS calculation
    fps_history = []
    fps_history_size = 10
    
    try:
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n✗ Error: Could not read frame from camera")
                print("Camera may have been disconnected or is being used by another app.")
                break
            
            # Detect objects
            detections = detector.detect(frame)
            
            # Print detected objects to console (for debugging)
            if detections:
                detected_names = [f"{d['class']} ({d['confidence']*100:.1f}%)" for d in detections]
                print(f"\rDetected: {', '.join(detected_names)}", end='', flush=True)
            
            # Draw detections
            draw_detections(frame, detections)
            
            # Calculate and display FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > fps_history_size:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Display stats
            stats_text = [
                f"FPS: {avg_fps:.1f}",
                f"Objects: {len(detections)}",
                f"Confidence: {args.confidence:.2f}"
            ]
            
            y_offset = 30
            for i, text in enumerate(stats_text):
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Show frame
            cv2.imshow('Object Detection', frame)
            
            # Save frame if recording
            if video_writer:
                video_writer.write(frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('c'):
                # Cycle confidence threshold
                thresholds = [0.3, 0.5, 0.7, 0.9]
                current_idx = thresholds.index(min(thresholds, key=lambda x: abs(x - args.confidence)))
                next_idx = (current_idx + 1) % len(thresholds)
                args.confidence = thresholds[next_idx]
                detector.confidence_threshold = args.confidence
                print(f"Confidence threshold changed to: {args.confidence}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()

