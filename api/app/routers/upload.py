"""
Image upload endpoints.
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

router = APIRouter()

# Directory for storing uploaded images
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file.
    
    Args:
        file: Image file to upload
        
    Returns:
        JSON response with the file info
    """
    # Generate a unique filename
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validate file extension
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Create a unique filename
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save the file
    try:
        # Read file contents
        contents = await file.read()
        
        # Write to file
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Open the image to verify it's valid and get dimensions
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                format = img.format
        except Exception as e:
            # Remove the file if it's not a valid image
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Return the file info
        return {
            "filename": unique_filename,
            "original_filename": file.filename,
            "content_type": file.content_type,
            "file_path": f"/uploads/{unique_filename}",
            "size": os.path.getsize(file_path),
            "width": width,
            "height": height,
            "format": format
        }
    
    except Exception as e:
        # Handle any errors
        if file_path.exists():
            os.remove(file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )


@router.get("/images")
async def list_images():
    """
    List all uploaded images.
    
    Returns:
        List of image info objects
    """
    try:
        images = []
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        format = img.format
                        
                        images.append({
                            "filename": file_path.name,
                            "file_path": f"/uploads/{file_path.name}",
                            "size": os.path.getsize(file_path),
                            "width": width,
                            "height": height,
                            "format": format
                        })
                except Exception:
                    # Skip invalid images
                    pass
        
        return {"images": images}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list images: {str(e)}"
        )


@router.delete("/images/{filename}")
async def delete_image(filename: str):
    """
    Delete an uploaded image.
    
    Args:
        filename: Name of the file to delete
        
    Returns:
        Success message
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image {filename} not found"
        )
    
    try:
        os.remove(file_path)
        return {"message": f"Image {filename} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete image: {str(e)}"
        )