"""
Aesthetic scoring module
Provides baseline aesthetic quality assessment
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List
import os
from pathlib import Path

class AestheticScorer:
    def __init__(self):
        """Initialize aesthetic scoring model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model for embeddings
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
        # Simple aesthetic scoring based on personal taste
        self.personal_taste_model = None
        self.load_personal_taste_model()
    
    def load_personal_taste_model(self):
        """Load or create personal taste model from strong/weak photos"""
        home_dir = Path.home()
        taste_model_path = str(home_dir / "pcb-data" / "models" / "personal_taste.pkl")
        
        if os.path.exists(taste_model_path):
            try:
                import pickle
                with open(taste_model_path, 'rb') as f:
                    self.personal_taste_model = pickle.load(f)
                print("Personal taste model loaded")
                return
            except Exception as e:
                print(f"Error loading personal taste model: {e}")
        
        # Create model from personal photos if available
        self.create_personal_taste_model()
    
    def create_personal_taste_model(self):
        """Create personal taste model from strong/weak photo collections"""
        home_dir = Path.home()
        strong_dir = home_dir / "pcb-data" / "personal" / "strong"
        weak_dir = home_dir / "pcb-data" / "personal" / "weak"
        
        if not strong_dir.exists() or not weak_dir.exists():
            print("Personal photo directories not found. Using default aesthetic scoring.")
            return
        
        # Get embeddings from strong and weak photos
        strong_embeddings = self._get_embeddings_from_dir(strong_dir)
        weak_embeddings = self._get_embeddings_from_dir(weak_dir)
        
        if len(strong_embeddings) > 0 and len(weak_embeddings) > 0:
            # Create simple k-NN based model
            from sklearn.neighbors import NearestNeighbors
            
            # Combine embeddings with labels
            all_embeddings = np.vstack([strong_embeddings, weak_embeddings])
            labels = np.array([1] * len(strong_embeddings) + [0] * len(weak_embeddings))
            
            # Train k-NN model
            knn = NearestNeighbors(n_neighbors=5, metric='cosine')
            knn.fit(all_embeddings)
            
            self.personal_taste_model = {
                'knn': knn,
                'labels': labels,
                'embeddings': all_embeddings
            }
            
            # Save the model
            home_dir = Path.home()
            models_dir = home_dir / "pcb-data" / "models"
            os.makedirs(models_dir, exist_ok=True)
            import pickle
            with open(models_dir / "personal_taste.pkl", 'wb') as f:
                pickle.dump(self.personal_taste_model, f)
            
            print(f"Personal taste model created from {len(strong_embeddings)} strong and {len(weak_embeddings)} weak photos")
        else:
            print("Not enough personal photos found for taste model")
    
    def _get_embeddings_from_dir(self, directory: Path) -> List[np.ndarray]:
        """Get CLIP embeddings from all images in a directory"""
        embeddings = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        for ext in image_extensions:
            for img_path in directory.glob(f"*{ext}"):
                try:
                    image = Image.open(img_path).convert("RGB")
                    embedding = self._get_clip_embedding(image)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        return embeddings
    
    def _get_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for an image"""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype("float32")[0]
    
    def score_aesthetic(self, image: Image.Image) -> Dict:
        """
        Score image aesthetic quality (1-10 scale)
        """
        # Get CLIP embedding
        embedding = self._get_clip_embedding(image)
        
        # Use personal taste model if available
        if self.personal_taste_model is not None:
            personal_score = self._score_with_personal_taste(embedding)
        else:
            personal_score = None
        
        # Use general aesthetic heuristics
        general_score = self._score_with_heuristics(image)
        
        # Combine scores
        if personal_score is not None:
            # Weight personal taste more heavily
            final_score = (personal_score * 0.7) + (general_score * 0.3)
        else:
            final_score = general_score
        
        return {
            "aesthetic_score": round(final_score, 1),
            "personal_score": round(personal_score, 1) if personal_score else None,
            "general_score": round(general_score, 1),
            "confidence": "high" if personal_score else "medium"
        }
    
    def _score_with_personal_taste(self, embedding: np.ndarray) -> float:
        """Score using personal taste model"""
        knn = self.personal_taste_model['knn']
        labels = self.personal_taste_model['labels']
        
        # Find nearest neighbors
        distances, indices = knn.kneighbors([embedding])
        
        # Weight by distance (closer = more important)
        weights = 1 / (distances[0] + 1e-6)  # Add small epsilon to avoid division by zero
        weights = weights / weights.sum()  # Normalize
        
        # Calculate weighted score
        neighbor_labels = labels[indices[0]]
        weighted_score = np.sum(weights * neighbor_labels)
        
        # Convert to 1-10 scale
        return 1 + (weighted_score * 9)
    
    def _score_with_heuristics(self, image: Image.Image) -> float:
        """Score using general photography heuristics"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Basic heuristics
        score = 5.0  # Start with neutral score
        
        # Color richness
        color_std = np.std(img_array)
        if color_std > 50:
            score += 1.0
        elif color_std < 20:
            score -= 1.0
        
        # Brightness balance
        brightness = np.mean(img_array)
        if 80 < brightness < 180:  # Good brightness range
            score += 0.5
        elif brightness < 50 or brightness > 200:
            score -= 1.0
        
        # Contrast
        contrast = np.std(img_array)
        if contrast > 40:
            score += 0.5
        elif contrast < 15:
            score -= 0.5
        
        # Ensure score is in 1-10 range
        return max(1.0, min(10.0, score))

# Global aesthetic scorer instance
aesthetic_scorer = AestheticScorer()
