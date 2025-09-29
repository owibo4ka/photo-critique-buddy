from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from typing import List
import shutil
from pathlib import Path
from PIL import Image
from .analysis_pipeline import pipeline

app = FastAPI(
    title="Photo Critique Buddy",
    description="AI-powered photo analysis and critique service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_image_file(file: UploadFile) -> bool:
    """Validate if the uploaded file is a valid image"""
    if not file.filename:
        return False
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return False
    
    # Check content type
    if not file.content_type or not file.content_type.startswith("image/"):
        return False
    
    return True

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Photo Critique Buddy API is running!"}

@app.post("/analyze")
async def analyze_photo(file: UploadFile = File(...)):
    """
    Upload and analyze a photo
    
    Args:
        file: Image file to analyze
        
    Returns:
        Analysis results
    """
    try:
        # Validate file
        if not validate_image_file(file):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a valid image file (jpg, jpeg, png, gif, bmp, tiff, webp)."
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size allowed is {MAX_FILE_SIZE // (1024*1024)}MB."
            )
        
        # Reset file pointer
        await file.seek(0)
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run the complete photo analysis pipeline
        try:
            # Load image for analysis
            image = Image.open(file_path).convert("RGB")
            
            # Run analysis
            analysis_result = pipeline.analyze_photo(image, file.filename)
            
            # Add file metadata
            analysis_result.update({
                "file_size": len(file_content),
                "content_type": file.content_type,
                "file_path": str(file_path),
                "status": "analyzed_successfully"
            })
            
        except Exception as analysis_error:
            # If analysis fails, still return basic info
            analysis_result = {
                "filename": file.filename,
                "file_size": len(file_content),
                "content_type": file.content_type,
                "status": "uploaded_but_analysis_failed",
                "error": str(analysis_error),
                "file_path": str(file_path)
            }
        
        return JSONResponse(content=analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file if it was created
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "photo-critique-buddy"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
