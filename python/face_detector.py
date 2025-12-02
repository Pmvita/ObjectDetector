#!/usr/bin/env python3
"""
Face Detection Module using OpenCV DNN.
Detects faces in images/frames using pre-trained DNN models.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class FaceDetector:
    """Face detector using OpenCV DNN with pre-trained models."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.load_model()
    
    def load_model(self):
        """Load face detection model using OpenCV DNN."""
        try:
            # Use OpenCV's built-in DNN face detector
            # This uses a pre-trained ResNet-based face detector
            model_dir = Path(__file__).parent / "models"
            model_dir.mkdir(exist_ok=True)
            
            # Try to use OpenCV's DNN face detector (works offline)
            # We'll use the OpenCV DNN module with a lightweight face detection model
            prototxt_path = model_dir / "deploy.prototxt"
            model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            
            # If model files don't exist, we'll use OpenCV's Haar Cascade as fallback
            if not prototxt_path.exists() or not model_path.exists():
                print("Face detection model files not found. Using Haar Cascade fallback...")
                self.use_haar_cascade = True
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    raise Exception("Could not load Haar Cascade classifier")
                print("✓ Haar Cascade face detector loaded")
                return
            
            try:
                self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.use_haar_cascade = False
                print("✓ DNN face detector loaded successfully")
            except Exception as e:
                print(f"Error loading DNN model: {e}. Using Haar Cascade fallback...")
                self.use_haar_cascade = True
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    raise Exception("Could not load Haar Cascade classifier")
                print("✓ Haar Cascade face detector loaded")
        except Exception as e:
            print(f"✗ Error loading face detection model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in frame.
        
        Returns:
            List of tuples: (x, y, width, height, confidence)
        """
        if self.use_haar_cascade:
            return self._detect_haar(frame)
        else:
            return self._detect_dnn(frame)
    
    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using DNN model."""
        if self.net is None:
            return []
        
        h, w = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            [104, 117, 123]  # Mean subtraction values
        )
        
        # Set input and forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                width = x2 - x1
                height = y2 - y1
                
                if width > 0 and height > 0:
                    faces.append((x1, y1, width, height, confidence))
        
        return faces
    
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using Haar Cascade (fallback)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in detected_faces:
            # Haar Cascade doesn't provide confidence, so we use a default value
            faces.append((x, y, w, h, 0.8))
        
        return faces
    
    def extract_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face region from frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, width, height)
        
        Returns:
            Extracted face region or None
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        
        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w > 0 and h > 0:
            return frame[y:y+h, x:x+w]
        return None

