# Real-Time Object Detection & Labeling

A real-time object detection application that uses your webcam to detect and label objects. Available in both web (TensorFlow.js + COCO-SSD) and Python (YOLO/COCO-SSD) versions.

## Features

- Real-time object detection from webcam feed
- Bounding boxes with labels and confidence scores
- Support for 80 object classes (COCO dataset)
- FPS monitoring and performance stats
- Modern, responsive UI (web version)
- Command-line interface (Python version)

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

5. Allow camera permissions when prompted

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
   python3 detector.py
   ```
   
   **Note:** The detector will automatically:
   - Try YOLO first (recommended - downloads model automatically)
   - Fall back to COCO-SSD if YOLO is not available
   - Use a simple detector if no ML models are available

5. Press `q` to quit, `s` to save a screenshot

## Requirements

### Web Version
- Modern web browser with camera support
- Node.js 16+ and npm

### Python Version
- Python 3.8+
- Webcam
- OpenCV, NumPy, and YOLO (Ultralytics) - see `python/requirements.txt`
- Optional: COCO-SSD model files (for fallback)

## Model Information

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
- Lower the video resolution in settings
- Reduce the frame processing rate
- Close other applications using the camera
