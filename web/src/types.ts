export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Detection {
  bbox: BoundingBox;
  class: string;
  score: number;
}

export interface DetectionResult {
  detections: Detection[];
  timestamp: number;
}

export interface ModelConfig {
  base?: 'mobilenet_v1' | 'mobilenet_v2' | 'lite_mobilenet_v2';
  modelUrl?: string;
}

export interface Stats {
  fps: number;
  objectCount: number;
  uniqueClasses: number;
  detectedObjects: Detection[];
}

// Face detection types (re-exported from faceTypes for convenience)
export type { FaceDetection, FaceAnalysis, FaceBoundingBox, FaceDetectionResult } from './faceTypes';

