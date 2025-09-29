# Photo Critique Buddy

AI-powered photo analysis and critique service built with FastAPI.

## Features

- **Image Upload**: Upload photos via the `/analyze` endpoint
- **File Validation**: Validates image file types and sizes
- **Error Handling**: Comprehensive error handling and validation
- **CORS Support**: Cross-origin resource sharing enabled
- **Health Checks**: Built-in health check endpoints

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   # Development mode with auto-reload
   python -m app.main
   
   # Or using uvicorn directly
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health

## API Endpoints

### POST /analyze
Upload and analyze a photo.

**Request**:
- Content-Type: `multipart/form-data`
- Body: Image file (jpg, jpeg, png, gif, bmp, tiff, webp)
- Max file size: 10MB

**Response**:
```json
{
  "filename": "photo.jpg",
  "file_size": 1024000,
  "content_type": "image/jpeg",
  "status": "uploaded_successfully",
  "message": "Photo uploaded successfully. Analysis coming soon!",
  "file_path": "uploads/photo.jpg"
}
```

### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "service": "photo-critique-buddy"
}
```

## Project Structure

```
photo-critique-buddy/
├── app/
│   ├── __init__.py
│   └── main.py          # Main FastAPI application
├── data/                # Data directories for analysis results
│   └── personal/
│       ├── maybe/
│       ├── strong/
│       └── weak/
├── models/              # ML models directory
├── uploads/             # Uploaded files (created automatically)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Development

The application includes:
- File type validation
- File size limits (10MB)
- Automatic upload directory creation
- Comprehensive error handling
- CORS middleware for web integration

## Next Steps

1. Add your photo analysis logic to the `/analyze` endpoint
2. Implement authentication if needed
3. Add database integration for storing analysis results
4. Configure proper CORS origins for production
5. Add logging and monitoring
