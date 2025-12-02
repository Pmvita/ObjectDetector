# Python Version - Real-Time Object Detection & Face Recognition

A Python-based real-time object detection and face recognition application using YOLO (recommended) or COCO-SSD models with OpenCV, plus face detection and analysis with DeepFace.

## Features

### Object Detection
- Real-time object detection from webcam
- 80 object classes from COCO dataset
- Bounding boxes with labels and confidence scores
- FPS monitoring
- Configurable confidence threshold
- Screenshot capture
- Video recording support

### Face Recognition & Analysis
- Real-time face detection from webcam
- Age estimation
- Gender classification
- Emotion detection (happy, sad, angry, surprise, fear, neutral, disgust)
- Race/ethnicity detection
- Bounding boxes with all detected attributes
- Performance optimized for real-time processing

## Requirements

- Python 3.8+
- Webcam
- OpenCV
- NumPy
- YOLO (Ultralytics) - **Recommended** - automatically downloads model
- COCO-SSD model files (optional fallback)
- DeepFace (for face analysis features)
- TensorFlow (required by DeepFace)

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   This installs OpenCV, NumPy, and YOLO (Ultralytics). YOLO will automatically download its model (~6MB) on first use - no manual download needed!

3. Run the detector:
   ```bash
   # For object detection (uses default webcam automatically):
   python3 detector.py
   
   # For face detection and analysis (uses default webcam automatically):
   python3 face_detector_app.py
   ```
   
   **Note:** Both applications automatically use your default webcam (camera 0). The webcam will start immediately when you run the application.
   
   **Object Detection:**
   - The detector automatically tries **YOLO first** (recommended - downloads model automatically)
   - Falls back to **COCO-SSD** if YOLO is not available
   - Uses a **simple detector** if no ML models are available (only shows "Object" labels)
   
   **To see actual object names:** Make sure YOLO is installed (`pip install ultralytics`). If you see "Object" labels, the ML model isn't loaded properly.
   
   **Face Detection:**
   - Uses OpenCV DNN face detector (with Haar Cascade fallback)
   - Analyzes faces for age, gender, emotion, and race using DeepFace
   - First run will download DeepFace models automatically (may take a few minutes)
   - Optimized for 50-60 FPS with background threading for analysis
   - Analysis runs every 20 frames by default (adjustable with `--analyze-interval`)
   - Use `--no-analysis` flag for maximum FPS (face detection only)

## Usage

### Object Detection

```bash
python detector.py
```

**Command-Line Options:**
- `--confidence FLOAT`: Confidence threshold (0.0-1.0, default: 0.5)
- `--camera INT`: Camera device index (default: 0)
- `--width INT`: Video width (default: 640)
- `--height INT`: Video height (default: 480)
- `--save PATH`: Save video to file (optional)

### Face Detection & Analysis

```bash
python face_detector_app.py
```

**Command-Line Options:**
- `--confidence FLOAT`: Face detection confidence threshold (0.0-1.0, default: 0.5)
- `--camera INT`: Camera device index (default: 0)
- `--width INT`: Video width (default: 640)
- `--height INT`: Video height (default: 480)
- `--analyze-interval INT`: Analyze faces every Nth frame (default: 20, higher = better FPS)
- `--target-fps INT`: Target FPS (default: 60, set to 30-50 if CPU can't keep up)
- `--no-analysis`: Disable face analysis for maximum FPS (only face detection, no age/gender/emotion/race)

**Examples:**

```bash
# Object detection with higher confidence
python detector.py --confidence 0.7

# Face detection with high FPS (50-60 FPS)
python face_detector_app.py --target-fps 60 --analyze-interval 30

# Maximum FPS mode (face detection only, no analysis)
python face_detector_app.py --no-analysis --target-fps 60

# Face detection with analysis every 10 frames
python face_detector_app.py --analyze-interval 10 --target-fps 50

# Use different camera
python face_detector_app.py --camera 1

# Custom resolution
python face_detector_app.py --width 1280 --height 720
```

## Controls

**Object Detection (detector.py):**
- **q**: Quit the application
- **s**: Save a screenshot
- **c**: Cycle through confidence thresholds (0.3, 0.5, 0.7, 0.9)

**Face Detection (face_detector_app.py):**
- **q**: Quit the application
- **s**: Save a screenshot

## Model Information

The detector supports multiple models (tries in order):

1. **YOLO (YOLOv8n) - Recommended**
   - Automatically downloads on first use (~6MB)
   - Fast and accurate
   - Detects 80 COCO classes
   - No manual setup required

2. **COCO-SSD (MobileNet-SSD) - Fallback**
   - Requires manual download (~23MB)
   - Run: `python3 models/download_models.py`
   - Detects 20 classes (subset of COCO)

3. **Simple Detector - Last Resort**
   - Basic motion detection
   - Only shows "Object" labels (no specific names)

**Detectable Objects (80 classes):**
- People, animals (cat, dog, bird, horse, etc.)
- Vehicles (car, bus, bicycle, motorcycle, train, etc.)
- Furniture (chair, couch, bed, dining table, etc.)
- Electronics (laptop, mouse, keyboard, cell phone, tv, etc.)
- Food items (banana, apple, pizza, bottle, cup, etc.)
- And many more...

## Performance Tips

### Face Detection Performance

**For 50-60 FPS:**
- Use `--target-fps 60` (default)
- Increase analysis interval: `--analyze-interval 20` or higher (default: 20)
- For maximum FPS: Use `--no-analysis` flag (face detection only, no age/gender/emotion/race)

**If CPU can't keep up:**
- Lower target FPS: `--target-fps 30` or `--target-fps 50`
- Increase analysis interval: `--analyze-interval 30` or `--analyze-interval 50`
- Lower resolution: `--width 320 --height 240`

**Examples:**
```bash
# High FPS with analysis
python3 face_detector_app.py --target-fps 60 --analyze-interval 30

# Maximum FPS (face detection only)
python3 face_detector_app.py --no-analysis --target-fps 60

# Balanced performance
python3 face_detector_app.py --target-fps 50 --analyze-interval 15
```

### General Performance Tips

- Use lower resolution for better FPS
- Adjust confidence threshold to filter detections
- Close other applications using the camera
- Use GPU acceleration if available (requires additional setup)

## Troubleshooting

### Camera Not Working

**First, run the camera diagnostic:**
```bash
python3 check_camera.py
```

**macOS Permission Issues (Most Common):**
1. Open **System Preferences** (or **System Settings** on macOS 13+)
2. Go to **Security & Privacy** > **Privacy** > **Camera**
3. Find **Terminal** (or **iTerm**, **Python**, etc.) in the list
4. **Check the box** to enable camera access
5. If Terminal is not in the list:
   - Run the detector once, then check again
   - Or reset camera permissions: `tccutil reset Camera`
   - Then restart Terminal

**Other Troubleshooting:**
- Close other apps using the camera (Zoom, FaceTime, Photo Booth, etc.)
- Try different camera indices: `python3 detector.py --camera 1`
- Try a different terminal app (iTerm2, VS Code terminal, PyCharm terminal)
- Restart your terminal/IDE
- Restart your computer if nothing else works

### Model Loading Issues / Seeing "Object" Instead of Names

**If you see "Object" labels instead of actual names:**

1. **Install YOLO (Recommended - Easiest):**
   ```bash
   pip install ultralytics
   ```
   Then restart the detector. YOLO will download its model automatically.

2. **Or use COCO-SSD (Alternative):**
   ```bash
   python3 models/download_models.py
   ```
   Note: COCO-SSD download may fail due to repository issues. YOLO is recommended.

3. **Check model loading:**
   - Look for "✓ Using YOLO model" or "✓ Using COCO-SSD model" in the output
   - If you see "⚠ WARNING: No ML models available!", install YOLO

**Other Issues:**
- Ensure you have internet connection (YOLO downloads model on first use)
- Check that model files are in the correct location (`python/models/`)
- Try reinstalling: `pip install --upgrade ultralytics`

### Low FPS

**For Face Detection:**
- Use `--no-analysis` flag for maximum FPS (face detection only)
- Increase `--analyze-interval` (e.g., `--analyze-interval 30` or `50`)
- Lower target FPS: `--target-fps 30` or `--target-fps 50`
- Lower resolution: `--width 320 --height 240`
- Example: `python3 face_detector_app.py --no-analysis --target-fps 60 --width 320 --height 240`

**General:**
- Lower the video resolution
- Increase confidence threshold to reduce detections
- Close other applications
- Use a more powerful computer or GPU

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

## Model Selection

The detector automatically selects the best available model:

1. **YOLO (Default)** - Installed via `requirements.txt`, downloads automatically
2. **COCO-SSD** - Fallback if YOLO unavailable, requires manual download
3. **Simple Detector** - Last resort, only shows "Object" labels

**To force a specific model:**
- YOLO: Just install `ultralytics` (already in requirements.txt)
- COCO-SSD: Download model files manually if YOLO fails

## Headless Mode

To run without display (save to file only):

```bash
python detector.py --save output.mp4
```

Then press 'q' to quit after recording.

