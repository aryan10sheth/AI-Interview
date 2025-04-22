import cv2
import numpy as np
from typing import Dict, Optional
import tempfile
import os

class VideoAnalyzer:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def analyze_video(self, video_path: str) -> Dict:
        """Analyze video for engagement metrics and body language."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Error opening video file")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            # Metrics to track
            face_detected_frames = 0
            eye_contact_frames = 0
            movement_data = []
            prev_face_center = None
            
            # Process frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    face_detected_frames += 1
                    
                    # Get primary face
                    x, y, w, h = faces[0]
                    face_roi_gray = gray[y:y+h, x:x+w]
                    
                    # Track face movement
                    current_face_center = (x + w//2, y + h//2)
                    if prev_face_center:
                        movement = self._calculate_movement(prev_face_center, current_face_center)
                        movement_data.append(movement)
                    prev_face_center = current_face_center
                    
                    # Detect eyes
                    eyes = self.eye_cascade.detectMultiScale(face_roi_gray)
                    if len(eyes) >= 2:
                        eye_contact_frames += 1
                        
            cap.release()
            
            # Calculate metrics
            face_detection_ratio = face_detected_frames / total_frames if total_frames > 0 else 0
            eye_contact_ratio = eye_contact_frames / face_detected_frames if face_detected_frames > 0 else 0
            movement_patterns = self._analyze_movement_patterns(movement_data)
            
            return {
                'face_detection_ratio': float(face_detection_ratio),
                'eye_contact_ratio': float(eye_contact_ratio),
                'movement_patterns': movement_patterns,
                'engagement_level': self._calculate_engagement_level(face_detection_ratio, eye_contact_ratio, movement_patterns),
                'duration': float(duration)
            }
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return {
                'face_detection_ratio': 0.0,
                'eye_contact_ratio': 0.0,
                'movement_patterns': 'insufficient_data',
                'engagement_level': 'medium',
                'duration': 0.0
            }
            
    def _calculate_movement(self, prev_center, current_center) -> float:
        """Calculate movement between two points."""
        return np.sqrt((current_center[0] - prev_center[0])**2 + 
                      (current_center[1] - prev_center[1])**2)
                      
    def _analyze_movement_patterns(self, movement_data: list) -> Dict:
        """Analyze movement patterns from collected data."""
        if not movement_data:
            return 'insufficient_data'
            
        movement_array = np.array(movement_data)
        
        # Calculate movement statistics
        avg_movement = np.mean(movement_array)
        movement_std = np.std(movement_array)
        
        # Classify movements
        stable_threshold = 10
        nodding_threshold = 20
        
        stable_ratio = np.mean(movement_array < stable_threshold)
        nodding_ratio = np.mean((movement_array >= stable_threshold) & 
                               (movement_array < nodding_threshold))
        side_to_side_ratio = np.mean(movement_array >= nodding_threshold)
        
        return {
            'stable': float(stable_ratio * 100),
            'nodding': float(nodding_ratio * 100),
            'side_to_side': float(side_to_side_ratio * 100),
            'average_movement': float(avg_movement),
            'movement_variation': float(movement_std)
        }
        
    def _calculate_engagement_level(self, face_ratio: float, eye_ratio: float, 
                                  movement_patterns: Dict) -> str:
        """Calculate overall engagement level."""
        if movement_patterns == 'insufficient_data':
            return 'medium'
            
        # Calculate engagement score
        engagement_score = (
            face_ratio * 0.3 +
            eye_ratio * 0.4 +
            (movement_patterns['stable'] / 100) * 0.3
        )
        
        # Convert score to level
        if engagement_score > 0.8:
            return 'high'
        elif engagement_score > 0.5:
            return 'medium'
        else:
            return 'low' 