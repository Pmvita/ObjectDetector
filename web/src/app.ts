import { ObjectDetector } from './detector';
import { FaceDetector } from './faceDetector';
import { FaceAnalyzer } from './faceAnalyzer';
import { CanvasRenderer } from './canvas';
import type { Detection, Stats, FaceDetection } from './types';

type DetectionMode = 'object' | 'face';

class ObjectDetectionApp {
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private detector: ObjectDetector;
  private faceDetector: FaceDetector;
  private faceAnalyzer: FaceAnalyzer;
  private renderer: CanvasRenderer;
  private isDetecting: boolean = false;
  private detectionMode: DetectionMode = 'object';
  private animationFrameId: number | null = null;
  private stats: Stats = {
    fps: 0,
    objectCount: 0,
    uniqueClasses: 0,
    detectedObjects: []
  };
  private faceDetections: FaceDetection[] = [];
  private analyzeFrameCount: number = 0;
  private readonly analyzeInterval: number = 5; // Analyze every 5th frame

  // UI Elements
  private statusEl: HTMLElement;
  private startBtn: HTMLButtonElement;
  private stopBtn: HTMLButtonElement;
  private screenshotBtn: HTMLButtonElement;
  private modeToggleBtn: HTMLButtonElement;
  private fpsEl: HTMLElement;
  private objectCountEl: HTMLElement;
  private uniqueClassesEl: HTMLElement;
  private objectsListEl: HTMLElement;

  // FPS calculation
  private lastFrameTime: number = 0;
  private fpsHistory: number[] = [];
  private readonly fpsHistorySize = 10;

  constructor() {
    this.video = document.getElementById('video') as HTMLVideoElement;
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement;
    this.statusEl = document.getElementById('status') as HTMLElement;
    this.startBtn = document.getElementById('startBtn') as HTMLButtonElement;
    this.stopBtn = document.getElementById('stopBtn') as HTMLButtonElement;
    this.screenshotBtn = document.getElementById('screenshotBtn') as HTMLButtonElement;
    this.modeToggleBtn = document.getElementById('modeToggleBtn') as HTMLButtonElement;
    this.fpsEl = document.getElementById('fps') as HTMLElement;
    this.objectCountEl = document.getElementById('objectCount') as HTMLElement;
    this.uniqueClassesEl = document.getElementById('uniqueClasses') as HTMLElement;
    this.objectsListEl = document.getElementById('objectsList') as HTMLElement;

    this.detector = new ObjectDetector();
    this.faceDetector = new FaceDetector();
    this.faceAnalyzer = new FaceAnalyzer();
    this.renderer = new CanvasRenderer(this.canvas, 0.5);

    this.setupEventListeners();
    this.initialize();
  }

  private setupEventListeners(): void {
    this.startBtn.addEventListener('click', () => this.startDetection());
    this.stopBtn.addEventListener('click', () => this.stopDetection());
    this.screenshotBtn.addEventListener('click', () => this.takeScreenshot());
    this.modeToggleBtn.addEventListener('click', () => this.toggleMode());
  }

  private async initialize(): Promise<void> {
    try {
      this.updateStatus('loading', 'Loading models...');
      
      // Load both detection models
      await Promise.all([
        this.detector.load(),
        this.faceDetector.load(),
        this.faceAnalyzer.load()
      ]);
      
      // Request camera access
      await this.setupCamera();
      
      this.updateStatus('ready', 'Starting detection...');
      this.startBtn.disabled = false;
      this.modeToggleBtn.disabled = false;
      
      // Auto-start detection
      this.startDetection();
    } catch (error) {
      console.error('Initialization error:', error);
      this.updateStatus('error', `Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async setupCamera(): Promise<void> {
    try {
      // Request access to default webcam
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'  // Use front-facing webcam by default
        }
      });

      this.video.srcObject = stream;
      
      // Wait for video metadata to load
      await new Promise((resolve) => {
        this.video.onloadedmetadata = () => {
          this.video.play();
          this.renderer.resize(this.video.videoWidth, this.video.videoHeight);
          resolve(null);
        };
      });
    } catch (error) {
      throw new Error(`Camera access denied or unavailable: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private startDetection(): void {
    if (this.isDetecting) return;
    
    this.isDetecting = true;
    this.startBtn.disabled = true;
    this.stopBtn.disabled = false;
    this.screenshotBtn.disabled = false;
    this.modeToggleBtn.disabled = false;
    
    const modeText = this.detectionMode === 'object' ? 'objects' : 'faces';
    this.updateStatus('ready', `Detecting ${modeText}...`);
    
    this.lastFrameTime = performance.now();
    this.detectLoop();
  }

  private stopDetection(): void {
    if (!this.isDetecting) return;
    
    this.isDetecting = false;
    this.startBtn.disabled = false;
    this.stopBtn.disabled = true;
    this.screenshotBtn.disabled = true;
    this.modeToggleBtn.disabled = false;
    this.updateStatus('ready', 'Detection stopped.');
    
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    this.renderer.clear();
    this.updateStatsDisplay();
  }

  private toggleMode(): void {
    if (this.isDetecting) {
      // Can't toggle while detecting
      return;
    }
    
    this.detectionMode = this.detectionMode === 'object' ? 'face' : 'object';
    const modeText = this.detectionMode === 'object' ? 'Object Detection' : 'Face Detection';
    this.modeToggleBtn.textContent = `Switch to ${this.detectionMode === 'object' ? 'Face' : 'Object'} Mode`;
    this.updateStatus('ready', `Mode: ${modeText}`);
  }

  private async detectLoop(): Promise<void> {
    if (!this.isDetecting) return;

    const currentTime = performance.now();
    const deltaTime = currentTime - this.lastFrameTime;
    this.lastFrameTime = currentTime;

    // Calculate FPS
    if (deltaTime > 0) {
      const fps = 1000 / deltaTime;
      this.fpsHistory.push(fps);
      if (this.fpsHistory.length > this.fpsHistorySize) {
        this.fpsHistory.shift();
      }
      const avgFps = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
      this.stats.fps = Math.round(avgFps);
    }

    try {
      if (this.detectionMode === 'object') {
        // Object detection mode
        const result = await this.detector.detect(this.video);
        
        // Update stats
        this.stats.detectedObjects = result.detections;
        this.stats.objectCount = result.detections.length;
        this.stats.uniqueClasses = new Set(result.detections.map(d => d.class)).size;

        // Render detections
        this.renderer.drawDetections(result.detections);
        
        // Update UI
        this.updateStatsDisplay();
        this.updateObjectsList(result.detections);
      } else {
        // Face detection mode
        const faceResult = await this.faceDetector.detect(this.video);
        
        // Analyze faces every Nth frame
        if (this.analyzeFrameCount % this.analyzeInterval === 0) {
          for (let i = 0; i < faceResult.faces.length; i++) {
            const face = faceResult.faces[i];
            const analysis = await this.faceAnalyzer.analyze(this.video, face.bbox);
            if (analysis) {
              face.analysis = analysis;
            }
          }
        }
        this.analyzeFrameCount++;
        
        // Update stored detections
        this.faceDetections = faceResult.faces;
        
        // Update stats
        this.stats.objectCount = faceResult.faces.length;
        this.stats.uniqueClasses = faceResult.faces.length; // Number of faces
        
        // Render face detections
        this.renderer.drawFaceDetections(faceResult.faces);
        
        // Update UI
        this.updateStatsDisplay();
        this.updateFacesList(faceResult.faces);
      }
    } catch (error) {
      console.error('Detection error:', error);
    }

    // Continue loop
    this.animationFrameId = requestAnimationFrame(() => this.detectLoop());
  }

  private updateStatsDisplay(): void {
    this.fpsEl.textContent = this.stats.fps.toString();
    this.objectCountEl.textContent = this.stats.objectCount.toString();
    this.uniqueClassesEl.textContent = this.stats.uniqueClasses.toString();
  }

  private updateObjectsList(detections: Detection[]): void {
    // Sort by confidence score (highest first)
    const sorted = [...detections].sort((a, b) => b.score - a.score);
    
    // Group by class
    const grouped = new Map<string, Detection[]>();
    sorted.forEach(detection => {
      const existing = grouped.get(detection.class) || [];
      existing.push(detection);
      grouped.set(detection.class, existing);
    });

    // Build HTML
    let html = '';
    grouped.forEach((detections, className) => {
      const count = detections.length;
      const avgConfidence = (detections.reduce((sum, d) => sum + d.score, 0) / count * 100).toFixed(1);
      
      html += `
        <div class="object-item">
          <span class="object-label">${className}${count > 1 ? ` (${count})` : ''}</span>
          <span class="object-confidence">${avgConfidence}%</span>
        </div>
      `;
    });

    this.objectsListEl.innerHTML = html || '<p style="color: #999; padding: 10px;">No objects detected</p>';
  }

  private updateFacesList(faces: FaceDetection[]): void {
    if (faces.length === 0) {
      this.objectsListEl.innerHTML = '<p style="color: #999; padding: 10px;">No faces detected</p>';
      return;
    }

    // Build HTML for faces
    let html = '';
    faces.forEach((face, index) => {
      const analysis = face.analysis;
      let details = '';
      
      if (analysis) {
        const parts: string[] = [];
        if (analysis.age !== undefined) parts.push(`Age: ${analysis.age}`);
        if (analysis.gender) parts.push(analysis.gender);
        if (analysis.emotion) parts.push(analysis.emotion);
        if (analysis.race) parts.push(analysis.race);
        details = parts.length > 0 ? ` - ${parts.join(', ')}` : '';
      }
      
      html += `
        <div class="object-item">
          <span class="object-label">Face ${index + 1}${details}</span>
          <span class="object-confidence">${(face.confidence * 100).toFixed(1)}%</span>
        </div>
      `;
    });

    this.objectsListEl.innerHTML = html;
  }

  private updateStatus(type: 'loading' | 'ready' | 'error', message: string): void {
    this.statusEl.className = `status ${type}`;
    this.statusEl.textContent = message;
  }

  private takeScreenshot(): void {
    if (!this.isDetecting) {
      alert('Please start detection first.');
      return;
    }

    const hasDetections = this.detectionMode === 'object' 
      ? this.stats.detectedObjects.length > 0
      : this.faceDetections.length > 0;

    if (!hasDetections) {
      alert(`Please wait for ${this.detectionMode === 'object' ? 'objects' : 'faces'} to be detected.`);
      return;
    }

    // Create a temporary canvas to combine video and detections
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = this.video.videoWidth;
    tempCanvas.height = this.video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    
    if (!tempCtx) return;

    // Draw video frame
    tempCtx.drawImage(this.video, 0, 0);
    
    // Draw detections based on mode
    if (this.detectionMode === 'object') {
      this.renderer.drawDetections(this.stats.detectedObjects);
    } else {
      this.renderer.drawFaceDetections(this.faceDetections);
    }
    tempCtx.drawImage(this.canvas, 0, 0);

    // Convert to blob and download
    const modePrefix = this.detectionMode === 'object' ? 'object-detection' : 'face-detection';
    tempCanvas.toBlob((blob) => {
      if (!blob) return;
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${modePrefix}-${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 'image/png');
  }

  dispose(): void {
    this.stopDetection();
    
    // Stop camera stream
    if (this.video.srcObject) {
      const stream = this.video.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
    
    this.detector.dispose();
    this.faceDetector.dispose();
    this.faceAnalyzer.dispose();
  }
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new ObjectDetectionApp();
  });
} else {
  new ObjectDetectionApp();
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  // Cleanup is handled by browser, but we can add explicit cleanup if needed
});

