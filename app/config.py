"""
Configuration settings for Photo Critique Buddy
"""

from pathlib import Path
import os

# Base directory for personal photos and models (outside the repo)
HOME_DIR = Path.home()
PHOTO_BASE_DIR = HOME_DIR / "pcb-data"

# Personal photo directories
PERSONAL_PHOTOS_DIR = PHOTO_BASE_DIR / "personal"
STRONG_PHOTOS_DIR = PERSONAL_PHOTOS_DIR / "strong"
MAYBE_PHOTOS_DIR = PERSONAL_PHOTOS_DIR / "maybe"
WEAK_PHOTOS_DIR = PERSONAL_PHOTOS_DIR / "weak"

# Model directories
MODELS_DIR = PHOTO_BASE_DIR / "models"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.pkl"
PERSONAL_TASTE_MODEL_PATH = MODELS_DIR / "personal_taste.pkl"

# Upload directory (inside repo, temporary)
UPLOAD_DIR = Path("uploads")

def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        STRONG_PHOTOS_DIR,
        MAYBE_PHOTOS_DIR,
        WEAK_PHOTOS_DIR,
        MODELS_DIR,
        UPLOAD_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories
