"""
Main analysis pipeline that combines all components
"""

from PIL import Image
from typing import Dict, List
from .aesthetic import aesthetic_scorer
from .critique import analyze_composition, generate_critique
from .embed_index import embedder
import os

class PhotoAnalysisPipeline:
    def __init__(self):
        """Initialize the complete analysis pipeline"""
        self.aesthetic_scorer = aesthetic_scorer
        self.embedder = embedder
        
        # Initialize embedder index if it doesn't exist
        if not self.embedder.load_index():
            print("Building FAISS index from personal photos...")
            self.embedder.build_index()
            self.embedder.save_index()
    
    def analyze_photo(self, image: Image.Image, filename: str = None) -> Dict:
        """
        Complete photo analysis pipeline
        
        Args:
            image: PIL Image object
            filename: Optional filename for logging
            
        Returns:
            Complete analysis results
        """
        print(f"Analyzing photo: {filename or 'unknown'}")
        
        # 1. Aesthetic scoring
        print("  → Computing aesthetic score...")
        aesthetic_results = self.aesthetic_scorer.score_aesthetic(image)
        
        # 2. Composition analysis
        print("  → Analyzing composition...")
        composition_results = analyze_composition(image)
        
        # 3. Find similar photos from personal collection
        print("  → Finding similar photos...")
        similar_photos = self.embedder.find_similar(image, k=5)
        
        # 4. Generate critique
        print("  → Generating critique...")
        critique_notes = generate_critique(composition_results, aesthetic_results["aesthetic_score"])
        
        # 5. Compile results
        results = {
            "filename": filename,
            "aesthetic": aesthetic_results["aesthetic_score"],
            "aesthetic_details": {
                "personal_score": aesthetic_results["personal_score"],
                "general_score": aesthetic_results["general_score"],
                "confidence": aesthetic_results["confidence"]
            },
            "composition": composition_results,
            "similar_photos": similar_photos,
            "critique": {
                "notes": critique_notes,
                "overall_rating": self._get_overall_rating(aesthetic_results["aesthetic_score"], composition_results),
                "strengths": self._identify_strengths(composition_results, aesthetic_results["aesthetic_score"]),
                "improvements": self._identify_improvements(composition_results, aesthetic_results["aesthetic_score"])
            }
        }
        
        print(f"  → Analysis complete! Score: {aesthetic_results['aesthetic_score']}/10")
        return results
    
    def _get_overall_rating(self, aesthetic_score: float, composition: Dict) -> str:
        """Get overall rating based on aesthetic and composition"""
        if aesthetic_score >= 8 and composition["thirds_score"] >= 0.6:
            return "Excellent"
        elif aesthetic_score >= 6 and composition["thirds_score"] >= 0.4:
            return "Good"
        elif aesthetic_score >= 4:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _identify_strengths(self, composition: Dict, aesthetic_score: float) -> List[str]:
        """Identify photo strengths"""
        strengths = []
        
        if aesthetic_score >= 7:
            strengths.append("Strong aesthetic appeal")
        
        if composition["thirds_score"] >= 0.6:
            strengths.append("Good compositional balance")
        
        if 0.3 <= composition["brightness_mean"] <= 0.7:
            strengths.append("Well-exposed image")
        
        if not composition["clipped_highlights"] and not composition["clipped_shadows"]:
            strengths.append("Good dynamic range")
        
        if composition["contrast_spread"] >= 0.3:
            strengths.append("Good contrast")
        
        return strengths
    
    def _identify_improvements(self, composition: Dict, aesthetic_score: float) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        if aesthetic_score < 5:
            improvements.append("Overall aesthetic quality could be enhanced")
        
        if composition["thirds_score"] < 0.4:
            improvements.append("Consider applying rule of thirds for better composition")
        
        if composition["brightness_mean"] < 0.3:
            improvements.append("Image appears underexposed")
        elif composition["brightness_mean"] > 0.7:
            improvements.append("Image may be overexposed")
        
        if composition["clipped_highlights"]:
            improvements.append("Recover clipped highlights")
        
        if composition["clipped_shadows"]:
            improvements.append("Lift clipped shadows")
        
        if composition["contrast_spread"] < 0.2:
            improvements.append("Increase contrast for more impact")
        
        return improvements

# Global pipeline instance
pipeline = PhotoAnalysisPipeline()
