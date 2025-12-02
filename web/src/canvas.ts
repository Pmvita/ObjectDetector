import type { Detection, FaceDetection } from './types';

export class CanvasRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private confidenceThreshold: number;

  constructor(canvas: HTMLCanvasElement, confidenceThreshold: number = 0.5) {
    this.canvas = canvas;
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Could not get 2D context from canvas');
    }
    this.ctx = context;
    this.confidenceThreshold = confidenceThreshold;
  }

  resize(videoWidth: number, videoHeight: number): void {
    this.canvas.width = videoWidth;
    this.canvas.height = videoHeight;
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  drawDetections(detections: Detection[]): void {
    this.clear();

    // Filter detections by confidence threshold
    const filteredDetections = detections.filter(
      detection => detection.score >= this.confidenceThreshold
    );

    filteredDetections.forEach(detection => {
      this.drawBoundingBox(detection);
      this.drawLabel(detection);
    });
  }

  private drawBoundingBox(detection: Detection): void {
    const { bbox, score } = detection;
    
    // Color based on confidence score
    const color = this.getColorForConfidence(score);
    
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 3;
    this.ctx.setLineDash([]);
    
    // Draw rectangle
    this.ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
    
    // Draw corner markers for better visibility
    const cornerSize = 10;
    this.ctx.fillStyle = color;
    
    // Top-left corner
    this.ctx.fillRect(bbox.x - 1, bbox.y - 1, cornerSize, 3);
    this.ctx.fillRect(bbox.x - 1, bbox.y - 1, 3, cornerSize);
    
    // Top-right corner
    this.ctx.fillRect(bbox.x + bbox.width - cornerSize + 1, bbox.y - 1, cornerSize, 3);
    this.ctx.fillRect(bbox.x + bbox.width - 2, bbox.y - 1, 3, cornerSize);
    
    // Bottom-left corner
    this.ctx.fillRect(bbox.x - 1, bbox.y + bbox.height - 2, cornerSize, 3);
    this.ctx.fillRect(bbox.x - 1, bbox.y + bbox.height - cornerSize + 1, 3, cornerSize);
    
    // Bottom-right corner
    this.ctx.fillRect(bbox.x + bbox.width - cornerSize + 1, bbox.y + bbox.height - 2, cornerSize, 3);
    this.ctx.fillRect(bbox.x + bbox.width - 2, bbox.y + bbox.height - cornerSize + 1, 3, cornerSize);
  }

  private drawLabel(detection: Detection): void {
    const { bbox, class: className, score } = detection;
    
    const label = `${className} ${(score * 100).toFixed(1)}%`;
    const padding = 8;
    const fontSize = 16;
    const fontFamily = 'Arial, sans-serif';
    
    this.ctx.font = `bold ${fontSize}px ${fontFamily}`;
    const textMetrics = this.ctx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = fontSize;
    
    const labelX = bbox.x;
    const labelY = bbox.y - 5;
    
    // Draw background rectangle for label
    const bgColor = this.getColorForConfidence(score);
    this.ctx.fillStyle = bgColor;
    this.ctx.fillRect(
      labelX - 2,
      labelY - textHeight - padding,
      textWidth + padding * 2,
      textHeight + padding
    );
    
    // Draw text
    this.ctx.fillStyle = '#FFFFFF';
    this.ctx.fillText(
      label,
      labelX + padding - 2,
      labelY - padding / 2
    );
  }

  private getColorForConfidence(score: number): string {
    // Green for high confidence (>0.7), yellow for medium (0.5-0.7), red for low (<0.5)
    if (score > 0.7) {
      return '#4CAF50'; // Green
    } else if (score > 0.5) {
      return '#FFC107'; // Yellow/Orange
    } else {
      return '#F44336'; // Red
    }
  }

  setConfidenceThreshold(threshold: number): void {
    this.confidenceThreshold = Math.max(0, Math.min(1, threshold));
  }

  captureScreenshot(): string {
    return this.canvas.toDataURL('image/png');
  }

  drawFaceDetections(faces: FaceDetection[]): void {
    this.clear();

    // Filter faces by confidence threshold
    const filteredFaces = faces.filter(
      face => face.confidence >= this.confidenceThreshold
    );

    filteredFaces.forEach(face => {
      this.drawFaceBoundingBox(face);
      this.drawFaceLabel(face);
    });
  }

  private drawFaceBoundingBox(face: FaceDetection): void {
    const { bbox, confidence } = face;
    
    // Color based on confidence score
    const color = this.getColorForConfidence(confidence);
    
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 3;
    this.ctx.setLineDash([]);
    
    // Draw rectangle
    this.ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
    
    // Draw corner markers for better visibility
    const cornerSize = 10;
    this.ctx.fillStyle = color;
    
    // Top-left corner
    this.ctx.fillRect(bbox.x - 1, bbox.y - 1, cornerSize, 3);
    this.ctx.fillRect(bbox.x - 1, bbox.y - 1, 3, cornerSize);
    
    // Top-right corner
    this.ctx.fillRect(bbox.x + bbox.width - cornerSize + 1, bbox.y - 1, cornerSize, 3);
    this.ctx.fillRect(bbox.x + bbox.width - 2, bbox.y - 1, 3, cornerSize);
    
    // Bottom-left corner
    this.ctx.fillRect(bbox.x - 1, bbox.y + bbox.height - 2, cornerSize, 3);
    this.ctx.fillRect(bbox.x - 1, bbox.y + bbox.height - cornerSize + 1, 3, cornerSize);
    
    // Bottom-right corner
    this.ctx.fillRect(bbox.x + bbox.width - cornerSize + 1, bbox.y + bbox.height - 2, cornerSize, 3);
    this.ctx.fillRect(bbox.x + bbox.width - 2, bbox.y + bbox.height - cornerSize + 1, 3, cornerSize);
  }

  private drawFaceLabel(face: FaceDetection): void {
    const { bbox, confidence, analysis } = face;
    
    const padding = 8;
    const fontSize = 14;
    const lineHeight = 18;
    const fontFamily = 'Arial, sans-serif';
    
    this.ctx.font = `bold ${fontSize}px ${fontFamily}`;
    
    // Build label lines
    const labelLines: string[] = [];
    labelLines.push(`Face ${(confidence * 100).toFixed(1)}%`);
    
    if (analysis) {
      if (analysis.age !== undefined) {
        labelLines.push(`Age: ${analysis.age}`);
      }
      if (analysis.gender) {
        labelLines.push(`Gender: ${analysis.gender}`);
      }
      if (analysis.emotion) {
        labelLines.push(`Emotion: ${analysis.emotion}`);
      }
      if (analysis.race) {
        labelLines.push(`Race: ${analysis.race}`);
      }
    }
    
    // Calculate maximum text width
    let maxWidth = 0;
    labelLines.forEach(line => {
      const metrics = this.ctx.measureText(line);
      maxWidth = Math.max(maxWidth, metrics.width);
    });
    
    const totalHeight = labelLines.length * lineHeight + padding * 2;
    const labelX = bbox.x;
    const labelY = bbox.y - 5;
    
    // Draw background rectangle for label
    const bgColor = this.getColorForConfidence(confidence);
    this.ctx.fillStyle = bgColor;
    this.ctx.fillRect(
      labelX - 2,
      labelY - totalHeight,
      maxWidth + padding * 2,
      totalHeight
    );
    
    // Draw text lines
    this.ctx.fillStyle = '#FFFFFF';
    labelLines.forEach((line, index) => {
      this.ctx.fillText(
        line,
        labelX + padding - 2,
        labelY - totalHeight + (index + 1) * lineHeight + padding / 2
      );
    });
  }
}

