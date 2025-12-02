#!/usr/bin/env python3
"""
Face Analysis Module using DeepFace.
Analyzes detected faces for age, gender, emotion, and race.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False
    print("Warning: DeepFace not installed. Install with: pip install deepface")


class FaceAnalyzer:
    """Face analyzer for age, gender, emotion, and race detection."""
    
    def __init__(self):
        self.is_available = HAS_DEEPFACE
        if not self.is_available:
            print("Face analysis features will be disabled. Install DeepFace to enable.")
    
    def analyze(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        Analyze face for age, gender, emotion, and race.
        
        Args:
            face_image: Face image (BGR format from OpenCV)
        
        Returns:
            Dictionary with analysis results or None if analysis fails
        """
        if not self.is_available:
            return None
        
        try:
            # DeepFace expects RGB format
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Run analysis with all required attributes
            result = DeepFace.analyze(
                face_rgb,
                actions=['age', 'gender', 'emotion', 'race'],
                enforce_detection=False,  # Don't fail if face detection fails
                silent=True  # Suppress verbose output
            )
            
            # DeepFace returns a list if multiple faces, or a dict for single face
            if isinstance(result, list):
                result = result[0] if result else None
            
            if result is None:
                return None
            
            # Extract and format results
            analysis = {
                'age': int(result.get('age', 0)),
                'gender': result.get('dominant_gender', 'Unknown'),
                'emotion': result.get('dominant_emotion', 'Unknown'),
                'race': result.get('dominant_race', 'Unknown'),
                'gender_confidence': result.get('gender', {}).get(result.get('dominant_gender', ''), 0.0),
                'emotion_confidence': result.get('emotion', {}).get(result.get('dominant_emotion', ''), 0.0),
                'race_confidence': result.get('race', {}).get(result.get('dominant_race', ''), 0.0)
            }
            
            return analysis
        except Exception as e:
            # Silently fail and return None - this allows detection to continue
            # even if analysis fails for a particular face
            return None
    
    def format_analysis_text(self, analysis: Dict) -> str:
        """
        Format analysis results as readable text.
        
        Args:
            analysis: Analysis dictionary from analyze()
        
        Returns:
            Formatted string
        """
        if not analysis:
            return "Analysis unavailable"
        
        parts = []
        
        # Age
        if 'age' in analysis:
            parts.append(f"Age: {analysis['age']}")
        
        # Gender
        if 'gender' in analysis:
            gender = analysis['gender']
            parts.append(f"Gender: {gender}")
        
        # Emotion
        if 'emotion' in analysis:
            emotion = analysis['emotion']
            parts.append(f"Emotion: {emotion}")
        
        # Race
        if 'race' in analysis:
            race = analysis['race']
            parts.append(f"Race: {race}")
        
        return " | ".join(parts)
    
    def is_ready(self) -> bool:
        """Check if analyzer is ready to use."""
        return self.is_available

