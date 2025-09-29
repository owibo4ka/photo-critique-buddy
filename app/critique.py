"""
Photo critique analysis module
Handles composition analysis, rule of thirds, brightness, etc.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import math

def analyze_composition(image: Image.Image) -> Dict:
    """
    Analyze photo composition including rule of thirds, brightness, contrast
    """
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Rule of thirds analysis
    thirds_score = analyze_rule_of_thirds(img_cv)
    
    # Brightness and contrast analysis
    brightness_analysis = analyze_brightness_contrast(img_cv)
    
    # Face detection for subject positioning
    face_analysis = detect_faces(img_cv)
    
    return {
        "thirds_score": thirds_score,
        "brightness_mean": brightness_analysis["mean"],
        "contrast_spread": brightness_analysis["contrast"],
        "clipped_highlights": brightness_analysis["clipped_highlights"],
        "clipped_shadows": brightness_analysis["clipped_shadows"],
        "face_detected": face_analysis["detected"],
        "face_position": face_analysis["position"]
    }

def analyze_rule_of_thirds(img: np.ndarray) -> float:
    """
    Analyze rule of thirds composition
    Returns score from 0-1 (1 = perfect thirds)
    """
    height, width = img.shape[:2]
    
    # Define thirds lines
    third_x1, third_x2 = width // 3, 2 * width // 3
    third_y1, third_y2 = height // 3, 2 * height // 3
    
    # Find most salient/important area (simplified - using center of mass of edges)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find center of mass of edges
    moments = cv2.moments(edges)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        # Fallback to image center
        cx, cy = width // 2, height // 2
    
    # Calculate distance to nearest thirds intersection
    thirds_points = [
        (third_x1, third_y1), (third_x2, third_y1),
        (third_x1, third_y2), (third_x2, third_y2)
    ]
    
    min_distance = float('inf')
    for px, py in thirds_points:
        distance = math.sqrt((cx - px)**2 + (cy - py)**2)
        min_distance = min(min_distance, distance)
    
    # Normalize by diagonal length
    diagonal = math.sqrt(width**2 + height**2)
    normalized_distance = min_distance / diagonal
    
    # Convert to score (closer to thirds = higher score)
    score = max(0, 1 - normalized_distance * 2)
    return round(score, 3)

def analyze_brightness_contrast(img: np.ndarray) -> Dict:
    """
    Analyze brightness and contrast
    """
    # Convert to LAB color space for better luminance analysis
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Normalize to 0-1
    l_normalized = l_channel.astype(np.float32) / 255.0
    
    # Calculate statistics
    mean_brightness = np.mean(l_normalized)
    std_contrast = np.std(l_normalized)
    
    # Check for clipping
    clipped_highlights = np.sum(l_channel > 250) / l_channel.size > 0.01  # >1% pixels
    clipped_shadows = np.sum(l_channel < 5) / l_channel.size > 0.01      # >1% pixels
    
    return {
        "mean": round(mean_brightness, 3),
        "contrast": round(std_contrast, 3),
        "clipped_highlights": clipped_highlights,
        "clipped_shadows": clipped_shadows
    }

def detect_faces(img: np.ndarray) -> Dict:
    """
    Simple face detection for subject positioning
    """
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Calculate face center relative to image
        face_center_x = (x + w/2) / img.shape[1]
        face_center_y = (y + h/2) / img.shape[0]
        
        return {
            "detected": True,
            "position": {
                "x": round(face_center_x, 3),
                "y": round(face_center_y, 3),
                "size": round((w * h) / (img.shape[0] * img.shape[1]), 3)
            }
        }
    
    return {"detected": False, "position": None}

def generate_critique(composition: Dict, aesthetic_score: float) -> List[str]:
    """
    Generate actionable critique based on analysis
    """
    notes = []
    
    # Aesthetic score feedback
    if aesthetic_score < 4:
        notes.append("The overall aesthetic could be improved. Consider adjusting composition or lighting.")
    elif aesthetic_score > 7:
        notes.append("Great aesthetic quality! This photo has strong visual appeal.")
    
    # Rule of thirds feedback
    if composition["thirds_score"] < 0.3:
        if composition["face_detected"] and composition["face_position"]:
            face_x = composition["face_position"]["x"]
            if 0.4 < face_x < 0.6:  # Face is centered
                notes.append("Subject is centered - try shifting toward a rule of thirds intersection for more dynamic composition.")
        else:
            notes.append("Consider applying the rule of thirds for more balanced composition.")
    
    # Brightness feedback
    if composition["brightness_mean"] < 0.3:
        notes.append("Image appears underexposed - consider increasing exposure or brightening in post.")
    elif composition["brightness_mean"] > 0.7:
        notes.append("Image may be overexposed - consider reducing exposure or recovering highlights.")
    
    # Clipping feedback
    if composition["clipped_highlights"]:
        notes.append("Highlights are clipped - try reducing exposure or using HDR techniques.")
    if composition["clipped_shadows"]:
        notes.append("Shadows are clipped - consider increasing exposure or using fill light.")
    
    # Contrast feedback
    if composition["contrast_spread"] < 0.2:
        notes.append("Low contrast - consider increasing contrast or using curves adjustment.")
    
    return notes
