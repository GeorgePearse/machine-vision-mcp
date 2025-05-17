"""
Claude API integration endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends

from ..claude_integration import ClaudeRequest, analyze_with_claude, analyze_with_openai

router = APIRouter()


@router.post("/claude/analyze")
async def claude_analyze(request: ClaudeRequest):
    """
    Analyze an image using Claude Vision API.
    
    Args:
        request: Claude API request parameters
        
    Returns:
        Claude's response
    """
    try:
        return await analyze_with_claude(request)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze with Claude: {str(e)}"
            )


@router.post("/openai/analyze")
async def openai_analyze(request: ClaudeRequest):
    """
    Analyze an image using OpenAI's Vision API.
    
    Args:
        request: API request parameters
        
    Returns:
        OpenAI's response
    """
    try:
        return await analyze_with_openai(request)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze with OpenAI: {str(e)}"
            )