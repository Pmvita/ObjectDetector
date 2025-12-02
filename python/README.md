# Python Version - Real-Time Object Detection

A Python-based real-time object detection application using YOLO (recommended) or COCO-SSD models with OpenCV.

## Features

- Real-time object detection from webcam
- 80 object classes from COCO dataset
- Bounding boxes with labels and confidence scores
- FPS monitoring
- Configurable confidence threshold
- Screenshot capture
- Video recording support

## Requirements

- Python 3.8+
- Webcam
- OpenCV
- NumPy
- YOLO (Ultralytics) - **Recommended** - automatically downloads model
- COCO-SSD model files (optional fallback)

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
   python3 detector.py
   ```
   
   **How it works:**
   - The detector automatically tries **YOLO first** (recommended - downloads model automatically)
   - Falls back to **COCO-SSD** if YOLO is not available
   - Uses a **simple detector** if no ML models are available (only shows "Object" labels)
   
   **To see actual object names:** Make sure YOLO is installed (`pip install ultralytics`). If you see "Object" labels, the ML model isn't loaded properly.

## Usage

### Basic Usage

```bash
python detector.py
```

### Command-Line Options

```bash
python detector.py --help
```

Options:
- `--confidence FLOAT`: Confidence threshold (0.0-1.0, default: 0.5)
- `--camera INT`: Camera device index (default: 0)
- `--width INT`: Video width (default: 640)
- `--height INT`: Video height (default: 480)
- `--save PATH`: Save video to file (optional)

### Examples

```bash
# Use higher confidence threshold
python detector.py --confidence 0.7

# Use different camera
python detector.py --camera 1

# Save video
python detector.py --save output.mp4

# Custom resolution
python detector.py --width 1280 --height 720
```

## Controls

While running:
- **q**: Quit the application
- **s**: Save a screenshot
- **c**: Cycle through confidence thresholds (0.3, 0.5, 0.7, 0.9)

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

