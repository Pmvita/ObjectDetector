# Real-Time Object Detection & Face Recognition

A real-time object detection and face recognition application that uses your webcam to detect objects and analyze faces. Available in both web (TensorFlow.js) and Python (YOLO/DeepFace) versions.

## Features

### Object Detection
- Real-time object detection from webcam feed
- Bounding boxes with labels and confidence scores
- Support for 80 object classes (COCO dataset)
- FPS monitoring and performance stats
- Modern, responsive UI (web version)
- Command-line interface (Python version)

### Face Recognition & Analysis
- Real-time face detection from webcam
- Age estimation
- Gender classification
- Emotion detection (happy, sad, angry, surprise, fear, neutral, disgust)
- Race/ethnicity detection
- Bounding boxes with all detected attributes
- Optimized for 50-60 FPS with background threading
- Toggle between object detection and face detection modes (web version)
- Configurable FPS target and analysis intervals

## Project Structure

```
ObjectDetector/
├── web/          # Web version (TensorFlow.js + COCO-SSD)
├── python/       # Python version (YOLO/COCO-SSD + OpenCV)
└── README.md     # This file
```

## Quick Start

### Web Version

1. Navigate to the web directory:
   ```bash
   cd web
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to the URL shown (usually `http://localhost:5173`)

5. Allow camera permissions when prompted - the webcam will start automatically

### Python Version

1. Navigate to the python directory:
   ```bash
   cd python
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
   
   This installs OpenCV, NumPy, and YOLO (Ultralytics). YOLO will automatically download its model on first use.

4. Run the detector:
   ```bash
   # For object detection (uses default webcam automatically):
   python3 detector.py
   
   # For face detection and analysis (uses default webcam automatically):
   python3 face_detector_app.py
   ```
   
   **Note:** Both applications automatically use your default webcam (camera 0). The webcam will start immediately when you run the application.
   
   **Object Detection:**
   - The detector will automatically:
     - Try YOLO first (recommended - downloads model automatically)
     - Fall back to COCO-SSD if YOLO is not available
     - Use a simple detector if no ML models are available
   
   **Face Detection:**
   - Uses OpenCV DNN face detector (with Haar Cascade fallback)
   - Analyzes faces using DeepFace (downloads models automatically on first run)
   - Optimized for 50-60 FPS with background threading for analysis
   - Analysis runs every 20 frames by default (adjustable with `--analyze-interval`)
   - Use `--no-analysis` flag for maximum FPS (face detection only)
   - Configurable target FPS with `--target-fps` option

5. Press `q` to quit, `s` to save a screenshot

## Requirements

### Web Version
- Modern web browser with camera support
- Node.js 16+ and npm
- TensorFlow.js models (load automatically)

### Python Version
- Python 3.8+
- Webcam
- OpenCV, NumPy, and YOLO (Ultralytics) - see `python/requirements.txt`
- DeepFace and TensorFlow (for face recognition features)
- Optional: COCO-SSD model files (for fallback)

## Model Information

### Object Detection Models

**Python Version:**
- **YOLO (Recommended)**: Uses YOLOv8n - automatically downloads on first use, detects 80 COCO classes
- **COCO-SSD (Fallback)**: MobileNet-SSD model - requires manual download if YOLO unavailable

**Web Version:**
- **COCO-SSD**: TensorFlow.js model - automatically loads from CDN

Both models can detect 80 different object classes including:
- People, animals (cat, dog, bird, etc.)
- Vehicles (car, bus, bicycle, motorcycle, etc.)
- Furniture (chair, couch, bed, etc.)
- Electronics (laptop, mouse, keyboard, etc.)
- Food items (banana, apple, pizza, etc.)
- And many more...

### Face Recognition Models

**Python Version:**
- **Face Detection**: OpenCV DNN face detector (with Haar Cascade fallback)
- **Face Analysis**: DeepFace library with pre-trained models for:
  - Age estimation (ApparentAge model)
  - Gender classification (Gender model)
  - Emotion detection (Ferc2013 model)
  - Race detection (Race model)
- Models download automatically on first use

**Web Version:**
- **Face Detection**: BlazeFace model via TensorFlow.js
- **Face Analysis**: Simplified analysis (can be extended with additional models)

## Troubleshooting

### Camera Not Working

**For Python version on macOS:**
1. Run the diagnostic script first:
   ```bash
   cd python
   python3 check_camera.py
   ```
2. Grant camera permissions:
   - System Preferences > Security & Privacy > Camera
   - Enable Terminal/Python in the list
3. If Terminal is not listed, try:
   - Running the detector once, then check again
   - Or use: `tccutil reset Camera` then restart Terminal

**General troubleshooting:**
- Ensure your browser/application has camera permissions
- Check that no other application is using the camera
- Try refreshing the page or restarting the Python script
- Close other apps using the camera (Zoom, FaceTime, etc.)

### Model Loading Issues

**Python Version:**
- **YOLO**: Will download automatically on first run (requires internet)
- **COCO-SSD**: If YOLO is not available, you can download COCO-SSD manually:
  ```bash
  python3 models/download_models.py
  ```
- If you see "Object" labels instead of actual names, install YOLO:
  ```bash
  pip install ultralytics
  ```

**Web Version:**
- Check browser console for errors
- Ensure internet connection (model loads from CDN)

### Performance Issues

**For Face Detection:**
- Use `--no-analysis` flag for maximum FPS (face detection only, no analysis)
- Increase `--analyze-interval` (e.g., `--analyze-interval 30`) for better FPS
- Lower target FPS if CPU can't keep up: `--target-fps 30`
- Lower resolution: `--width 320 --height 240`

**General:**
- Lower the video resolution in settings
- Reduce the frame processing rate
- Close other applications using the camera
