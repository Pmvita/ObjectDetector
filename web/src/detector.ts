import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import type { Detection, DetectionResult, ModelConfig } from './types';

export class ObjectDetector {
  private model: cocoSsd.ObjectDetection | null = null;
  private isModelLoaded: boolean = false;
  private config: ModelConfig;

  constructor(config: ModelConfig = {}) {
    this.config = config;
  }

  async load(): Promise<void> {
    try {
      // Initialize TensorFlow.js backend
      await tf.ready();
      
      // Load COCO-SSD model
      this.model = await cocoSsd.load({
        base: this.config.base || 'mobilenet_v2',
        modelUrl: this.config.modelUrl
      });
      
      this.isModelLoaded = true;
      console.log('COCO-SSD model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      throw new Error(`Failed to load COCO-SSD model: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async detect(videoElement: HTMLVideoElement): Promise<DetectionResult> {
    if (!this.isModelLoaded || !this.model) {
      throw new Error('Model not loaded. Call load() first.');
    }

    if (videoElement.readyState !== videoElement.HAVE_ENOUGH_DATA) {
      return {
        detections: [],
        timestamp: Date.now()
      };
    }

    try {
      const predictions = await this.model.detect(videoElement);
      
      const detections: Detection[] = predictions.map(pred => ({
        bbox: {
          x: pred.bbox[0],
          y: pred.bbox[1],
          width: pred.bbox[2],
          height: pred.bbox[3]
        },
        class: pred.class,
        score: pred.score
      }));

      return {
        detections,
        timestamp: Date.now()
      };
    } catch (error) {
      console.error('Error during detection:', error);
      return {
        detections: [],
        timestamp: Date.now()
      };
    }
  }

  isLoaded(): boolean {
    return this.isModelLoaded;
  }

  dispose(): void {
    if (this.model) {
      // TensorFlow.js models don't have explicit dispose, but we can clear the reference
      this.model = null;
      this.isModelLoaded = false;
    }
  }
}

