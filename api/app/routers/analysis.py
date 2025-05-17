"""
Image analysis endpoints using Machine Vision MCP.
"""
import base64
import io
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from skimage import color, filters, io as skio, measure, morphology, segmentation

# Import the MCP
from src.mcp import MachineLearningMCP

router = APIRouter()

# Initialize MCP
mcp = MachineLearningMCP()

# Directory for uploaded images
UPLOAD_DIR = Path("./uploads")


class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""
    CORNER_DETECTION = "corner_detection"
    BLOB_DETECTION = "blob_detection"
    TEMPLATE_MATCHING = "template_matching"
    HOG = "hog"
    DAISY = "daisy"
    LBP = "lbp"
    BRIEF = "brief"
    SIFT = "sift"
    GABOR = "gabor"
    OBJECT_SEGMENTATION = "object_segmentation"
    PIXEL_COUNT = "pixel_count"
    AREA_MEASUREMENT = "area_measurement"


class AnalysisRequest(BaseModel):
    """Request model for image analysis."""
    filename: str = Field(..., description="Filename of the uploaded image")
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    template_filename: Optional[str] = Field(None, description="Filename of the template image (for template matching)")
    threshold: Optional[float] = Field(0.5, description="Threshold value for segmentation")
    parameters: Optional[Dict] = Field({}, description="Additional parameters for the analysis")


class PixelAnalysisRequest(BaseModel):
    """Request model for pixel analysis."""
    filename: str = Field(..., description="Filename of the uploaded image")
    color_ranges: List[Dict[str, List[int]]] = Field(..., description="List of color ranges to count pixels for")
    label: Optional[str] = Field(None, description="Label for the counted pixels")


def get_image_path(filename: str) -> Path:
    """
    Get the full path to an uploaded image.
    
    Args:
        filename: Name of the image file
        
    Returns:
        Full path to the image
        
    Raises:
        HTTPException: If the image does not exist
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image {filename} not found"
        )
    
    return file_path


def image_to_base64(image):
    """Convert image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@router.post("/analyze")
async def analyze_image(request: AnalysisRequest):
    """
    Analyze an image using the specified method.
    
    Args:
        request: Analysis request parameters
        
    Returns:
        Analysis results including visualizations
    """
    image_path = get_image_path(request.filename)
    
    try:
        # Different analysis based on the type
        if request.analysis_type == AnalysisType.CORNER_DETECTION:
            method = request.parameters.get("method", "harris")
            result = mcp.detect_corners(
                image_path=str(image_path),
                method=method,
                visualize=True,
                **request.parameters
            )
            
            return {
                "corners": result.corners,
                "scores": result.scores,
                "visualization": result.visualization,
                "count": len(result.corners),
                "message": f"Detected {len(result.corners)} corners using {method} method"
            }
            
        elif request.analysis_type == AnalysisType.BLOB_DETECTION:
            min_sigma = float(request.parameters.get("min_sigma", 1.0))
            max_sigma = float(request.parameters.get("max_sigma", 30.0))
            
            result = mcp.detect_blobs(
                image_path=str(image_path),
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                visualize=True,
                **request.parameters
            )
            
            return {
                "blobs": result.blobs,
                "visualization": result.visualization,
                "count": len(result.blobs),
                "message": f"Detected {len(result.blobs)} blobs"
            }
            
        elif request.analysis_type == AnalysisType.TEMPLATE_MATCHING:
            if not request.template_filename:
                raise HTTPException(
                    status_code=400,
                    detail="Template filename is required for template matching"
                )
                
            template_path = get_image_path(request.template_filename)
            method = request.parameters.get("method", "ncc")
            
            result = mcp.detect_template(
                image_path=str(image_path),
                template_path=str(template_path),
                method=method,
                visualize=True,
                **request.parameters
            )
            
            return {
                "matches": result.matches,
                "scores": result.scores,
                "best_match": result.best_match,
                "best_score": result.best_score,
                "visualization": result.visualization,
                "count": len(result.matches),
                "message": f"Found {len(result.matches)} matches using {method} method"
            }
            
        elif request.analysis_type == AnalysisType.HOG:
            orientations = int(request.parameters.get("orientations", 9))
            
            result = mcp.detect_hog(
                image_path=str(image_path),
                orientations=orientations,
                visualize=True,
                **request.parameters
            )
            
            return {
                "feature_length": len(result.features),
                "visualization": result.visualization,
                "message": f"Extracted HOG features with {len(result.features)} dimensions"
            }
            
        elif request.analysis_type == AnalysisType.DAISY:
            step = int(request.parameters.get("step", 4))
            radius = int(request.parameters.get("radius", 15))
            
            result = mcp.detect_daisy(
                image_path=str(image_path),
                step=step,
                radius=radius,
                visualize=True,
                **request.parameters
            )
            
            return {
                "keypoint_count": len(result.keypoints),
                "descriptor_length": len(result.descriptors[0]) if result.descriptors else 0,
                "visualization": result.visualization,
                "message": f"Extracted DAISY features at {len(result.keypoints)} keypoints"
            }
            
        elif request.analysis_type == AnalysisType.LBP:
            radius = int(request.parameters.get("radius", 3))
            n_points = int(request.parameters.get("n_points", 24))
            
            result = mcp.detect_lbp(
                image_path=str(image_path),
                radius=radius,
                n_points=n_points,
                visualize=True,
                **request.parameters
            )
            
            return {
                "histogram_bins": len(result.histogram),
                "visualization": result.visualization,
                "message": f"Created LBP histogram with {len(result.histogram)} bins"
            }
            
        elif request.analysis_type == AnalysisType.OBJECT_SEGMENTATION:
            # Load image
            image = skio.imread(image_path)
            
            # Convert to grayscale if needed
            if image.ndim == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image)
            else:
                gray = image
                
            # Apply threshold
            threshold = request.threshold
            binary = gray > threshold
            
            # Remove small objects
            min_size = int(request.parameters.get("min_size", 50))
            binary_cleaned = morphology.remove_small_objects(binary, min_size=min_size)
            
            # Label regions
            labeled, num_labels = measure.label(binary_cleaned, return_num=True)
            
            # Get region properties
            regions = measure.regionprops(labeled)
            
            # Calculate areas
            areas = [region.area for region in regions]
            centroids = [region.centroid for region in regions]
            bboxes = [region.bbox for region in regions]
            
            # Create visualization
            # Convert labeled image to RGB
            labeled_rgb = color.label2rgb(labeled, image=gray, bg_label=0)
            
            # Encode image to base64
            pil_image = Image.fromarray((labeled_rgb * 255).astype(np.uint8))
            vis_base64 = image_to_base64(pil_image)
            
            return {
                "object_count": num_labels,
                "areas": areas,
                "centroids": centroids,
                "bboxes": bboxes,
                "visualization": vis_base64,
                "message": f"Detected {num_labels} objects"
            }
            
        elif request.analysis_type == AnalysisType.PIXEL_COUNT:
            # Load image
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Get image dimensions
            height, width = image_array.shape[:2]
            total_pixels = height * width
            
            # Convert to RGB if not already
            if image_array.ndim == 2:
                # Grayscale image
                rgb_image = np.stack([image_array] * 3, axis=-1)
            else:
                rgb_image = image_array[:, :, :3]
            
            # Count non-white pixels (assuming white is background)
            non_white = np.sum(np.any(rgb_image < 245, axis=2))
            
            # Create a binary mask of non-white pixels
            mask = np.any(rgb_image < 245, axis=2)
            
            # Create visualization
            vis_image = image.copy()
            draw = ImageDraw.Draw(vis_image)
            
            # Highlight non-white pixels
            for y in range(height):
                for x in range(width):
                    if mask[y, x]:
                        draw.point((x, y), fill=(255, 0, 0, 128))
            
            vis_base64 = image_to_base64(vis_image)
            
            return {
                "total_pixels": total_pixels,
                "counted_pixels": int(non_white),
                "percentage": float(non_white) / total_pixels * 100,
                "visualization": vis_base64,
                "message": f"Image contains {non_white} non-white pixels out of {total_pixels} total pixels ({non_white/total_pixels*100:.2f}%)"
            }
            
        elif request.analysis_type == AnalysisType.AREA_MEASUREMENT:
            # Load image
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Get image dimensions
            height, width = image_array.shape[:2]
            total_area = height * width
            
            # Convert to RGB if not already
            if image_array.ndim == 2:
                # Grayscale image
                rgb_image = np.stack([image_array] * 3, axis=-1)
            else:
                rgb_image = image_array[:, :, :3]
            
            # Use threshold to detect objects
            threshold = request.threshold
            if rgb_image.ndim == 3:
                gray = color.rgb2gray(rgb_image)
            else:
                gray = rgb_image
                
            binary = gray > threshold
            
            # Remove small objects
            min_size = int(request.parameters.get("min_size", 50))
            binary_cleaned = morphology.remove_small_objects(binary, min_size=min_size)
            
            # Calculate area in pixels
            object_area = np.sum(binary_cleaned)
            
            # Optional: Convert to real-world units if scale is provided
            pixels_per_unit = float(request.parameters.get("pixels_per_unit", 1.0))
            unit = request.parameters.get("unit", "pixel")
            
            real_area = object_area / (pixels_per_unit ** 2)
            
            # Create visualization
            # Overlay binary mask on original image
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            for y in range(height):
                for x in range(width):
                    if binary_cleaned[y, x]:
                        draw.point((x, y), fill=(0, 255, 0, 128))
            
            composite = Image.alpha_composite(image.convert('RGBA'), overlay)
            vis_base64 = image_to_base64(composite)
            
            return {
                "total_area_pixels": total_area,
                "object_area_pixels": int(object_area),
                "object_area_units": float(real_area),
                "unit": unit,
                "percentage": float(object_area) / total_area * 100,
                "visualization": vis_base64,
                "message": f"Object area: {real_area:.2f} {unit}² ({object_area} pixels, {object_area/total_area*100:.2f}% of image)"
            }
            
        else:
            # For other analysis types, use the corresponding MCP method
            method_name = f"detect_{request.analysis_type}"
            if hasattr(mcp, method_name):
                result = getattr(mcp, method_name)(
                    image_path=str(image_path),
                    visualize=True,
                    **request.parameters
                )
                
                return {
                    "result": result.dict(),
                    "visualization": result.visualization,
                    "message": f"Analysis completed using {request.analysis_type}"
                }
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported analysis type: {request.analysis_type}"
                )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/analyze/pixels")
async def analyze_pixels(request: PixelAnalysisRequest):
    """
    Count pixels matching specific color ranges.
    
    Args:
        request: Pixel analysis request
        
    Returns:
        Pixel count results
    """
    image_path = get_image_path(request.filename)
    
    try:
        # Load image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Get image dimensions
        if image_array.ndim == 2:
            height, width = image_array.shape
            # Convert grayscale to RGB
            image_array = np.stack([image_array] * 3, axis=-1)
        else:
            height, width, _ = image_array.shape
            
        total_pixels = height * width
        
        # Process each color range
        results = []
        
        for i, color_range in enumerate(request.color_ranges):
            label = color_range.get("label", f"Color {i+1}")
            
            # Get color ranges
            r_range = color_range.get("r", [0, 255])
            g_range = color_range.get("g", [0, 255])
            b_range = color_range.get("b", [0, 255])
            
            # Create mask for pixels in the color range
            r_mask = (image_array[:, :, 0] >= r_range[0]) & (image_array[:, :, 0] <= r_range[1])
            g_mask = (image_array[:, :, 1] >= g_range[0]) & (image_array[:, :, 1] <= g_range[1])
            b_mask = (image_array[:, :, 2] >= b_range[0]) & (image_array[:, :, 2] <= b_range[1])
            
            color_mask = r_mask & g_mask & b_mask
            
            # Count pixels
            pixel_count = np.sum(color_mask)
            
            # Create visualization
            vis_image = image.copy()
            draw = ImageDraw.Draw(vis_image)
            
            for y in range(height):
                for x in range(width):
                    if color_mask[y, x]:
                        # Highlight matching pixels
                        draw.point((x, y), fill=(255, 0, 0, 128))
            
            vis_base64 = image_to_base64(vis_image)
            
            results.append({
                "label": label,
                "color_range": {
                    "r": r_range,
                    "g": g_range,
                    "b": b_range
                },
                "pixel_count": int(pixel_count),
                "percentage": float(pixel_count) / total_pixels * 100,
                "visualization": vis_base64
            })
        
        return {
            "total_pixels": total_pixels,
            "results": results,
            "message": f"Analyzed pixel colors in {request.filename}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pixel analysis failed: {str(e)}"
        )