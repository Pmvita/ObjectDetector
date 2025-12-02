export interface FaceBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceAnalysis {
  age?: number;
  gender?: string;
  emotion?: string;
  race?: string;
  genderConfidence?: number;
  emotionConfidence?: number;
  raceConfidence?: number;
}

export interface FaceDetection {
  bbox: FaceBoundingBox;
  confidence: number;
  analysis?: FaceAnalysis;
}

export interface FaceDetectionResult {
  faces: FaceDetection[];
  timestamp: number;
}

