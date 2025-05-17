"""
Tests for the utility functions of the Machine Vision MCP.
"""
import base64
import os

import numpy as np
import pytest
from PIL import Image
from skimage import data

from src.features.utils import (
    image_to_base64,
    normalize_image,
    preprocess_image,
    visualize_features,
)


def test_preprocess_image():
    """Test image preprocessing."""
    # Test with grayscale image
    gray_image = data.camera()
    processed = preprocess_image(gray_image)
    
    assert processed.ndim == 2
    assert processed.min() >= 0
    assert processed.max() <= 1.0
    assert processed.shape == gray_image.shape
    
    # Test with color image
    color_image = data.astronaut()
    processed = preprocess_image(color_image)
    
    assert processed.ndim == 2
    assert processed.min() >= 0
    assert processed.max() <= 1.0
    assert processed.shape == (color_image.shape[0], color_image.shape[1])


def test_normalize_image():
    """Test image normalization."""
    # Create a test image with arbitrary values
    image = np.random.random((100, 100)) * 5 - 2  # Range: -2 to 3
    
    normalized = normalize_image(image)
    
    assert normalized.dtype == np.uint8
    assert normalized.min() >= 0
    assert normalized.max() <= 255
    assert normalized.shape == image.shape


def test_image_to_base64():
    """Test image to base64 conversion."""
    # Test with different image formats
    
    # Grayscale image
    gray_image = data.camera()
    base64_gray = image_to_base64(gray_image)
    
    assert isinstance(base64_gray, str)
    assert len(base64_gray) > 0
    
    # Color image
    color_image = data.astronaut()
    base64_color = image_to_base64(color_image)
    
    assert isinstance(base64_color, str)
    assert len(base64_color) > 0
    
    # Float image (0-1)
    float_image = np.random.random((50, 50))
    base64_float = image_to_base64(float_image)
    
    assert isinstance(base64_float, str)
    assert len(base64_float) > 0


def test_visualize_features():
    """Test feature visualization."""
    image = data.camera()
    
    # Test with corners
    corners = [(50, 50), (100, 100), (150, 150)]
    vis_corners = visualize_features(image, corners, "corners")
    
    assert isinstance(vis_corners, str)
    assert len(vis_corners) > 0
    
    # Test with blobs
    blobs = [(50, 50, 10), (100, 100, 15), (150, 150, 20)]
    vis_blobs = visualize_features(image, blobs, "blobs")
    
    assert isinstance(vis_blobs, str)
    assert len(vis_blobs) > 0
    
    # Test with keypoints
    keypoints = [(50, 50), (100, 100), (150, 150)]
    vis_keypoints = visualize_features(image, keypoints, "orb")
    
    assert isinstance(vis_keypoints, str)
    assert len(vis_keypoints) > 0
    
    # Test with Haar features
    haar_features = [(20, 20, 30, 30), (60, 60, 40, 40)]
    vis_haar = visualize_features(image, haar_features, "haar")
    
    assert isinstance(vis_haar, str)
    assert len(vis_haar) > 0