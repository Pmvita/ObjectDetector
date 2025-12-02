import { ObjectDetector } from './detector';
import { CanvasRenderer } from './canvas';
import type { Detection, Stats } from './types';

class ObjectDetectionApp {
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private detector: ObjectDetector;
  private renderer: CanvasRenderer;
  private isDetecting: boolean = false;
  private animationFrameId: number | null = null;
  private stats: Stats = {
    fps: 0,
    objectCount: 0,
    uniqueClasses: 0,
    detectedObjects: []
  };

  // UI Elements
  private statusEl: HTMLElement;
  private startBtn: HTMLButtonElement;
  private stopBtn: HTMLButtonElement;
  private screenshotBtn: HTMLButtonElement;
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
    this.fpsEl = document.getElementById('fps') as HTMLElement;
    this.objectCountEl = document.getElementById('objectCount') as HTMLElement;
    this.uniqueClassesEl = document.getElementById('uniqueClasses') as HTMLElement;
    this.objectsListEl = document.getElementById('objectsList') as HTMLElement;

    this.detector = new ObjectDetector();
    this.renderer = new CanvasRenderer(this.canvas, 0.5);

    this.setupEventListeners();
    this.initialize();
  }

  private setupEventListeners(): void {
    this.startBtn.addEventListener('click', () => this.startDetection());
    this.stopBtn.addEventListener('click', () => this.stopDetection());
    this.screenshotBtn.addEventListener('click', () => this.takeScreenshot());
  }

  private async initialize(): Promise<void> {
    try {
      this.updateStatus('loading', 'Loading model...');
      
      // Load the detection model
      await this.detector.load();
      
      // Request camera access
      await this.setupCamera();
      
      this.updateStatus('ready', 'Starting detection...');
      this.startBtn.disabled = false;
      
      // Auto-start detection
      this.startDetection();
    } catch (error) {
      console.error('Initialization error:', error);
      this.updateStatus('error', `Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async setupCamera(): Promise<void> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
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
    this.updateStatus('ready', 'Detecting objects...');
    
    this.lastFrameTime = performance.now();
    this.detectLoop();
  }

  private stopDetection(): void {
    if (!this.isDetecting) return;
    
    this.isDetecting = false;
    this.startBtn.disabled = false;
    this.stopBtn.disabled = true;
    this.screenshotBtn.disabled = true;
    this.updateStatus('ready', 'Detection stopped.');
    
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    this.renderer.clear();
    this.updateStatsDisplay();
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
      // Perform detection
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

  private updateStatus(type: 'loading' | 'ready' | 'error', message: string): void {
    this.statusEl.className = `status ${type}`;
    this.statusEl.textContent = message;
  }

  private takeScreenshot(): void {
    if (!this.isDetecting || this.stats.detectedObjects.length === 0) {
      alert('Please start detection first and wait for objects to be detected.');
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
    
    // Draw detections
    this.renderer.drawDetections(this.stats.detectedObjects);
    tempCtx.drawImage(this.canvas, 0, 0);

    // Convert to blob and download
    tempCanvas.toBlob((blob) => {
      if (!blob) return;
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `object-detection-${Date.now()}.png`;
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

