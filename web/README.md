# Web Version - Real-Time Object Detection

A browser-based real-time object detection application using TensorFlow.js and the COCO-SSD model.

## Features

- Real-time object detection from webcam
- 80 object classes from COCO dataset
- Bounding boxes with labels and confidence scores
- FPS monitoring
- Detection statistics
- Screenshot capture
- Modern, responsive UI

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open your browser to the URL shown (usually `http://localhost:5173`)

4. Allow camera permissions when prompted

5. Detection will start automatically once the model loads

## Usage

- **Start Detection**: Begin real-time object detection (auto-starts by default)
- **Stop Detection**: Pause detection
- **Take Screenshot**: Capture current frame with detections

## Controls

The application will automatically:
- Request camera access
- Load the COCO-SSD model
- Start detection automatically
- Display detected objects with bounding boxes and labels in real-time
- Show FPS and detection statistics

## Model Information

The COCO-SSD model can detect 80 different object classes including:
- People, animals, vehicles
- Furniture, electronics
- Food items, sports equipment
- And many more...

## Performance Tips

- Close other applications using the camera
- Use a modern browser (Chrome, Firefox, Edge)
- Ensure good lighting for better detection
- Lower video resolution if experiencing performance issues

## Browser Compatibility

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: May require additional permissions

## Troubleshooting

### Camera Not Working
- Check browser permissions for camera access
- Ensure no other application is using the camera
- Try refreshing the page

### Model Loading Issues
- Check browser console for errors
- Ensure stable internet connection (model may need to download)
- Try clearing browser cache

### Low FPS
- Close other browser tabs
- Lower video resolution
- Reduce number of objects in view

