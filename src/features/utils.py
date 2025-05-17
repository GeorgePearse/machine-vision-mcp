"""
Utility functions for the Machine Vision MCP.
"""
import base64
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.transform import resize


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for feature extraction.

    Args:
        image: Input image as a numpy array

    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale if needed
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = rgb2gray(image)
    else:
        gray = image
    
    # Ensure proper scaling (0-1 range)
    if gray.max() > 1.0:
        gray = gray / 255.0
    
    return gray


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for visualization.

    Args:
        image: Input image as a numpy array

    Returns:
        Normalized image suitable for visualization
    """
    # Convert to proper range for visualization
    normalized = rescale_intensity(image, out_range=(0, 1))
    
    # Convert to 8-bit
    normalized = (normalized * 255).astype(np.uint8)
    
    return normalized


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert an image array to base64 string.

    Args:
        image: Image as a numpy array

    Returns:
        Base64 encoded string of the image
    """
    # Ensure image is in proper format for PIL
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = normalize_image(image)
    
    # Convert to PIL Image
    if image.ndim == 2:
        pil_image = Image.fromarray(image, mode='L')
    elif image.ndim == 3 and image.shape[2] == 1:
        pil_image = Image.fromarray(image.squeeze(), mode='L')
    elif image.ndim == 3 and image.shape[2] == 3:
        pil_image = Image.fromarray(image, mode='RGB')
    elif image.ndim == 3 and image.shape[2] == 4:
        pil_image = Image.fromarray(image, mode='RGBA')
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Save to BytesIO and convert to base64
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return encoded


def visualize_features(
    image: np.ndarray, 
    features: Union[List[Tuple], np.ndarray], 
    feature_type: str
) -> str:
    """
    Visualize detected features on an image.

    Args:
        image: Input image
        features: List of feature coordinates
        feature_type: Type of features ('corners', 'blobs', etc.)

    Returns:
        Base64 encoded visualization image
    """
    # Convert image to proper format for PIL
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            vis_image = (image * 255).astype(np.uint8)
        else:
            vis_image = image.astype(np.uint8)
    else:
        vis_image = image.copy()
    
    # Convert to RGB if grayscale
    if vis_image.ndim == 2:
        vis_image = np.stack([vis_image] * 3, axis=-1)
    elif vis_image.ndim == 3 and vis_image.shape[2] == 1:
        vis_image = np.concatenate([vis_image] * 3, axis=-1)
    
    # Create PIL image for drawing
    pil_image = Image.fromarray(vis_image)
    draw = ImageDraw.Draw(pil_image)
    
    # Draw features based on type
    if feature_type == "corners":
        # Draw corners as small crosses
        for feature in features:
            y, x = feature[:2]
            size = 5
            draw.line([(x-size, y), (x+size, y)], fill=(255, 0, 0), width=2)
            draw.line([(x, y-size), (x, y+size)], fill=(255, 0, 0), width=2)
    
    elif feature_type == "blobs":
        # Draw blobs as circles
        for feature in features:
            y, x = feature[:2]
            radius = feature[2] if len(feature) > 2 else 5
            draw.ellipse(
                [(x-radius, y-radius), (x+radius, y+radius)], 
                outline=(255, 0, 0), width=2
            )
    
    elif feature_type in ["orb", "brief", "sift"]:
        # Draw keypoints as circles with orientation
        for feature in features:
            if hasattr(feature, 'x') and hasattr(feature, 'y'):
                # KeyPoint object
                x, y = feature.x, feature.y
                size = feature.size if hasattr(feature, 'size') else 5
                angle = feature.angle if hasattr(feature, 'angle') else None
            else:
                # Tuple or array
                y, x = feature[:2]
                size = feature[2] if len(feature) > 2 else 5
                angle = None
            
            # Draw keypoint circle
            draw.ellipse(
                [(x-size/2, y-size/2), (x+size/2, y+size/2)], 
                outline=(0, 255, 0), width=2
            )
            
            # Draw orientation line if available
            if angle is not None:
                end_x = x + size * np.cos(angle)
                end_y = y + size * np.sin(angle)
                draw.line([(x, y), (end_x, end_y)], fill=(0, 0, 255), width=2)
    
    elif feature_type in ["daisy", "peaks", "template"]:
        # Draw simple markers
        for feature in features:
            y, x = feature[:2]
            size = 3
            draw.rectangle(
                [(x-size, y-size), (x+size, y+size)], 
                outline=(0, 255, 255), width=2
            )
    
    elif feature_type == "haar":
        # Draw rectangles for Haar features
        for feature in features:
            y, x, h, w = feature
            draw.rectangle(
                [(x, y), (x+w, y+h)], 
                outline=(255, 255, 0), width=2
            )
    
    # Convert to base64
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return encoded