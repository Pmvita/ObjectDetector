import * as blazeface from '@tensorflow-models/blazeface';
import * as tf from '@tensorflow/tfjs';
import type { FaceDetection, FaceDetectionResult } from './faceTypes';

export class FaceDetector {
  private model: blazeface.BlazeFaceModel | null = null;
  private isModelLoaded: boolean = false;

  async load(): Promise<void> {
    try {
      // Initialize TensorFlow.js backend
      await tf.ready();
      
      // Load BlazeFace model
      this.model = await blazeface.load({
        maxFaces: 10,
        iouThreshold: 0.3,
        scoreThreshold: 0.5
      });
      
      this.isModelLoaded = true;
      console.log('BlazeFace model loaded successfully');
    } catch (error) {
      console.error('Error loading BlazeFace model:', error);
      throw new Error(`Failed to load BlazeFace model: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async detect(videoElement: HTMLVideoElement): Promise<FaceDetectionResult> {
    if (!this.isModelLoaded || !this.model) {
      throw new Error('Model not loaded. Call load() first.');
    }

    if (videoElement.readyState !== videoElement.HAVE_ENOUGH_DATA) {
      return {
        faces: [],
        timestamp: Date.now()
      };
    }

    try {
      const predictions = await this.model.estimateFaces(videoElement, false);
      
      const faces: FaceDetection[] = predictions.map(pred => {
        const start = pred.topLeft as [number, number];
        const end = pred.bottomRight as [number, number];
        
        return {
          bbox: {
            x: start[0],
            y: start[1],
            width: end[0] - start[0],
            height: end[1] - start[1]
          },
          confidence: pred.probability[0]
        };
      });

      return {
        faces,
        timestamp: Date.now()
      };
    } catch (error) {
      console.error('Error during face detection:', error);
      return {
        faces: [],
        timestamp: Date.now()
      };
    }
  }

  isLoaded(): boolean {
    return this.isModelLoaded;
  }

  dispose(): void {
    if (this.model) {
      this.model = null;
      this.isModelLoaded = false;
    }
  }
}

