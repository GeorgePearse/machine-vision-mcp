"""
Claude API integration for the Machine Vision MCP.
"""
import os
import base64
import json
from io import BytesIO
from typing import Dict, List, Optional, Union

from fastapi import HTTPException, UploadFile
from pydantic import BaseModel

# Anthropic API client (install with: pip install anthropic)
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# Optional: Import OpenAI for comparison/alternative
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .routers.analysis import get_image_path

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Models
class ClaudeRequest(BaseModel):
    """Request model for Claude API."""
    filename: str
    prompt: str
    system_prompt: Optional[str] = None
    model: str = "claude-3-opus-20240229"
    max_tokens: int = 4096


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for API requests."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode image: {str(e)}"
        )


async def analyze_with_claude(request: ClaudeRequest) -> Dict:
    """
    Analyze an image using Claude Vision.
    
    Args:
        request: Claude API request parameters
        
    Returns:
        Claude's response
    """
    if not Anthropic:
        raise HTTPException(
            status_code=500,
            detail="Anthropic Python package not installed. Run: pip install anthropic"
        )
        
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable not set"
        )
    
    # Get image path
    image_path = get_image_path(request.filename)
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(str(image_path))
    
    try:
        # Initialize Claude client
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Default system prompt if not provided
        system_prompt = request.system_prompt or (
            "You are a computer vision assistant with expertise in image analysis. "
            "You have access to scikit-image tools to analyze the provided image. "
            "Respond with detailed observations and measurements about the image."
        )
        
        # Create Claude API request
        response = client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {
                            "type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        )
        
        # For demonstration purposes, we'll include how the response should be handled
        # In a real application, Claude would use the API to get specific measurements
        
        # Extract the content from Claude's response
        content = response.content[0].text
        
        return {
            "role": "assistant",
            "content": content,
            "model": request.model,
            "image_info": {
                "filename": request.filename,
                "path": f"/uploads/{request.filename}"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Claude API error: {str(e)}"
        )


async def analyze_with_openai(request: ClaudeRequest) -> Dict:
    """
    Alternative implementation using OpenAI's GPT-4 Vision.
    
    Args:
        request: API request parameters
        
    Returns:
        OpenAI's response
    """
    if not OpenAI:
        raise HTTPException(
            status_code=500,
            detail="OpenAI Python package not installed. Run: pip install openai"
        )
        
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not set"
        )
    
    # Get image path
    image_path = get_image_path(request.filename)
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(str(image_path))
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Default system prompt if not provided
        system_prompt = request.system_prompt or (
            "You are a computer vision assistant with expertise in image analysis. "
            "Respond with detailed observations and measurements about the image."
        )
        
        # Create OpenAI API request
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            max_tokens=request.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract content from response
        content = response.choices[0].message.content
        
        return {
            "role": "assistant",
            "content": content,
            "model": "gpt-4-vision-preview",
            "image_info": {
                "filename": request.filename,
                "path": f"/uploads/{request.filename}"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {str(e)}"
        )