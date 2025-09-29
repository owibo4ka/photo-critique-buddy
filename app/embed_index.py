"""
Image embedding and similarity search using CLIP and FAISS
"""

import os
import pickle
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from typing import List, Dict, Tuple
from pathlib import Path

class ImageEmbedder:
    def __init__(self):
        """Initialize CLIP model for image embeddings"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
        # FAISS index
        self.index = None
        self.image_paths = []
        self.dimension = 512  # CLIP ViT-B/32 embedding dimension
        
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP embedding for a single image"""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            # Normalize embeddings
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype("float32")
    
    def build_index(self, data_dir: str = None) -> None:
        """Build FAISS index from personal strong photos"""
        if data_dir is None:
            # Use default location outside the repo
            home_dir = Path.home()
            data_dir = str(home_dir / "pcb-data" / "personal" / "strong")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Warning: {data_dir} does not exist. Creating empty index.")
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            return
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(data_path.glob(f"*{ext}"))
            image_files.extend(data_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {data_dir}")
            self.index = faiss.IndexFlatIP(self.dimension)
            return
        
        print(f"Building index from {len(image_files)} images...")
        
        embeddings = []
        self.image_paths = []
        
        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
                embedding = self.embed_image(image)
                embeddings.append(embedding[0])  # Remove batch dimension
                self.image_paths.append(str(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings_array)
            print(f"Index built with {len(embeddings)} embeddings")
        else:
            print("No valid embeddings created")
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def find_similar(self, query_image: Image.Image, k: int = 5) -> List[Dict]:
        """Find k most similar images from the index"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.embed_image(query_image)
        
        # Search for similar images
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_paths):
                results.append({
                    "path": self.image_paths[idx],
                    "similarity": float(score),
                    "filename": os.path.basename(self.image_paths[idx])
                })
        
        return results
    
    def save_index(self, index_path: str = None) -> None:
        """Save the FAISS index and image paths"""
        if index_path is None:
            # Use default location outside the repo
            home_dir = Path.home()
            index_path = str(home_dir / "pcb-data" / "models" / "faiss_index.pkl")
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss_path = index_path.replace('.pkl', '.faiss')
        faiss.write_index(self.index, faiss_path)
        
        # Save metadata
        with open(index_path, 'wb') as f:
            pickle.dump({
                'image_paths': self.image_paths,
                'dimension': self.dimension
            }, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str = None) -> bool:
        """Load the FAISS index and image paths"""
        if index_path is None:
            # Use default location outside the repo
            home_dir = Path.home()
            index_path = str(home_dir / "pcb-data" / "models" / "faiss_index.pkl")
        
        try:
            faiss_path = index_path.replace('.pkl', '.faiss')
            
            if not os.path.exists(faiss_path) or not os.path.exists(index_path):
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(faiss_path)
            
            # Load metadata
            with open(index_path, 'rb') as f:
                metadata = pickle.load(f)
                self.image_paths = metadata['image_paths']
                self.dimension = metadata['dimension']
            
            print(f"Index loaded with {len(self.image_paths)} images")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

# Global embedder instance
embedder = ImageEmbedder()
