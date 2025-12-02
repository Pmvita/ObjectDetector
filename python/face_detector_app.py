#!/usr/bin/env python3
"""
Real-Time Face Detection and Analysis Application.
Detects faces and analyzes age, gender, emotion, and race.
"""

import cv2
import numpy as np
import argparse
import time
import sys
import threading
from pathlib import Path
from queue import Queue

from face_detector import FaceDetector
from face_analyzer import FaceAnalyzer


class FaceDetectionApp:
    """Main application for face detection and analysis."""
    
    def __init__(self, confidence_threshold: float = 0.5, analyze_interval: int = 20, target_fps: int = 60):
        self.confidence_threshold = confidence_threshold
        self.analyze_interval = analyze_interval  # Analyze every Nth frame
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps  # Target time per frame
        self.frame_count = 0
        
        # Initialize detectors
        print("Initializing face detector...")
        self.face_detector = FaceDetector(confidence_threshold=confidence_threshold)
        
        print("Initializing face analyzer...")
        self.face_analyzer = FaceAnalyzer()
        
        # Store analysis results for each face
        self.face_analyses = {}  # Key: face_id (based on position), Value: analysis dict
        
        # Threading for async analysis
        self.analysis_queue = Queue(maxsize=2)  # Limit queue size to prevent memory buildup
        self.analysis_thread = None
        self.analysis_running = False
    
    def draw_face_detection(self, frame: np.ndarray, faces: list, analyses: dict = None):
        """
        Draw face detection results on frame.
        
        Args:
            frame: Input frame
            faces: List of detected faces (x, y, width, height, confidence)
            analyses: Dictionary mapping face positions to analysis results
        """
        for i, (x, y, w, h, confidence) in enumerate(faces):
            # Generate face ID based on position (simple approach)
            face_id = f"{x}_{y}_{w}_{h}"
            
            # Get analysis if available
            analysis = analyses.get(face_id) if analyses else None
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Prepare label text
            label_parts = [f"Face {confidence*100:.1f}%"]
            
            if analysis:
                # Add analysis results
                if 'age' in analysis:
                    label_parts.append(f"Age: {analysis['age']}")
                if 'gender' in analysis:
                    gender = analysis['gender']
                    label_parts.append(f"{gender}")
                if 'emotion' in analysis:
                    emotion = analysis['emotion']
                    label_parts.append(f"{emotion}")
                if 'race' in analysis:
                    race = analysis['race']
                    label_parts.append(f"{race}")
            
            # Draw label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Calculate text size for all lines
            line_height = 20
            y_offset = y - 10
            
            for idx, text in enumerate(label_parts):
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    frame,
                    (x - 1, y_offset - text_height - baseline - 5),
                    (x + text_width + 10, y_offset + 5),
                    (0, 0, 0),  # Black border
                    2
                )
                cv2.rectangle(
                    frame,
                    (x, y_offset - text_height - baseline - 4),
                    (x + text_width + 9, y_offset + 4),
                    color,
                    -1  # Filled
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    text,
                    (x + 5, y_offset - baseline),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    thickness
                )
                
                y_offset -= (line_height + 5)
    
    def _analysis_worker(self):
        """Background thread worker for face analysis."""
        while self.analysis_running:
            try:
                # Get task from queue (with timeout to allow checking running flag)
                task = self.analysis_queue.get(timeout=0.1)
                if task is None:
                    break
                
                face_id, face_roi = task
                if face_roi is not None and face_roi.size > 0:
                    analysis = self.face_analyzer.analyze(face_roi)
                    if analysis:
                        self.face_analyses[face_id] = analysis
                
                self.analysis_queue.task_done()
            except:
                continue
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect faces and optionally analyze them.
        
        Args:
            frame: Input frame
        
        Returns:
            Frame with detections drawn
        """
        # Detect faces (this is fast)
        faces = self.face_detector.detect(frame)
        
        # Queue faces for analysis in background thread (every Nth frame)
        if self.face_analyzer.is_ready() and self.frame_count % self.analyze_interval == 0:
            for i, (x, y, w, h, confidence) in enumerate(faces):
                # Ensure minimum size for analysis
                if w >= 50 and h >= 50:
                    face_id = f"{x}_{y}_{w}_{h}"
                    face_roi = self.face_detector.extract_face(frame, (x, y, w, h))
                    
                    # Try to add to queue (non-blocking)
                    try:
                        self.analysis_queue.put_nowait((face_id, face_roi))
                    except:
                        pass  # Queue full, skip this frame
        
        # Use stored analyses for drawing (fallback to previous if new analysis not available)
        current_analyses = {}
        for x, y, w, h, _ in faces:
            face_id = f"{x}_{y}_{w}_{h}"
            # Try to find matching analysis (allow some position tolerance)
            best_match = None
            for stored_id, stored_analysis in self.face_analyses.items():
                stored_parts = stored_id.split('_')
                if len(stored_parts) == 4:
                    stored_x, stored_y, stored_w, stored_h = map(int, stored_parts)
                    # Check if positions are similar (within 50 pixels)
                    if (abs(stored_x - x) < 50 and abs(stored_y - y) < 50 and
                        abs(stored_w - w) < 30 and abs(stored_h - h) < 30):
                        best_match = stored_analysis
                        break
            
            if best_match:
                current_analyses[face_id] = best_match
        
        # Draw detections
        self.draw_face_detection(frame, faces, current_analyses)
        
        self.frame_count += 1
        return frame
    
    def run(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Run the face detection application.
        
        Args:
            camera_index: Camera device index (default: 0 for default webcam)
            width: Video width
            height: Video height
        """
        # Initialize video capture - use webcam by default
        if camera_index == 0:
            print("Opening default webcam (camera 0)...")
        else:
            print(f"Opening camera {camera_index}...")
        
        import platform
        if platform.system() == 'Darwin':
            cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open camera {camera_index}")
            sys.exit(1)
        
        # Set video properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Test camera
        print("Testing camera...")
        ret, test_frame = None, None
        for attempt in range(5):
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                break
            time.sleep(0.2)
        
        if not ret or test_frame is None:
            print("✗ Error: Camera opened but could not read frames")
            cap.release()
            sys.exit(1)
        
        print(f"✓ Camera working! Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"Target FPS: {self.target_fps}")
        print(f"Analysis interval: Every {self.analyze_interval} frames")
        print("\nControls:")
        print("  Press 'q' to quit")
        print("  Press 's' to save screenshot")
        print("\nStarting face detection...\n")
        
        # Start background analysis thread (only if analysis is enabled)
        if self.face_analyzer.is_ready() and self.analyze_interval < 999999:
            self.analysis_running = True
            self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
            self.analysis_thread.start()
            print("Background analysis thread started")
        elif self.analyze_interval >= 999999:
            print("Face analysis disabled - maximum FPS mode")
        
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
                    break
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Calculate and display FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_history.append(fps)
                if len(fps_history) > fps_history_size:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)
                
                # Frame rate limiting to target FPS
                sleep_time = self.frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Display stats
                num_faces = len([f for f in self.face_analyses.keys()])
                stats_text = [
                    f"FPS: {avg_fps:.1f}",
                    f"Faces: {num_faces}",
                    f"Confidence: {self.confidence_threshold:.2f}"
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
                cv2.imshow('Face Detection & Analysis', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"face_detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Stop analysis thread
            if self.analysis_running:
                self.analysis_running = False
                try:
                    self.analysis_queue.put_nowait(None)  # Signal thread to stop
                except:
                    pass
                if self.analysis_thread:
                    self.analysis_thread.join(timeout=1.0)
            
            cap.release()
            cv2.destroyAllWindows()
            print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description='Real-Time Face Detection & Analysis')
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
        help='Camera device index (default: 0 for default webcam)'
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
        '--analyze-interval',
        type=int,
        default=20,
        help='Analyze faces every Nth frame (default: 20, higher = better FPS but slower analysis updates)'
    )
    parser.add_argument(
        '--target-fps',
        type=int,
        default=60,
        help='Target FPS (default: 60, set lower if CPU can\'t keep up)'
    )
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help='Disable face analysis (age/gender/emotion/race) for maximum FPS - only detect faces'
    )
    
    args = parser.parse_args()
    
    # If analysis disabled, set interval to very high value
    analyze_interval = 999999 if args.no_analysis else args.analyze_interval
    
    app = FaceDetectionApp(
        confidence_threshold=args.confidence,
        analyze_interval=analyze_interval,
        target_fps=args.target_fps
    )
    app.run(
        camera_index=args.camera,
        width=args.width,
        height=args.height
    )


if __name__ == "__main__":
    main()

