"""
Tests for the detection features of the Machine Vision MCP.
"""
import base64
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage import data, io

from src.features.detection import (
    detect_blobs,
    detect_brief,
    detect_censure,
    detect_corners,
    detect_daisy,
    detect_fisher_vectors,
    detect_gabor,
    detect_haar,
    detect_hog,
    detect_lbp,
    detect_orb,
    detect_peaks,
    detect_sift,
    detect_template,
)
from src.features.utils import preprocess_image, normalize_image, image_to_base64


# Create a test directory for saving test images
TEST_DIR = Path("tests/test_images")
TEST_DIR.mkdir(exist_ok=True, parents=True)


@pytest.fixture
def test_image():
    """Create a test image for testing."""
    # Use skimage's sample image
    image = data.camera()
    
    # Save the test image to test_images directory
    test_path = TEST_DIR / "test_image.png"
    io.imsave(test_path, image)
    
    return image, str(test_path)


@pytest.fixture
def test_template():
    """Create a small template for testing template matching."""
    # Use a small region from the camera image
    image = data.camera()
    template = image[100:150, 100:150]
    
    # Save the template to test_images directory
    test_path = TEST_DIR / "test_template.png"
    io.imsave(test_path, template)
    
    return template, str(test_path)


def test_corners_detection(test_image):
    """Test corner detection."""
    image, _ = test_image
    
    # Test with different methods
    for method in ["harris", "shi_tomasi", "kitchen_rosenfeld"]:
        result = detect_corners(image, method=method, visualize=True)
        
        # Basic validation
        assert "corners" in result.dict()
        assert "scores" in result.dict()
        assert len(result.corners) > 0
        assert len(result.scores) == len(result.corners)
        
        # Check visualization
        assert result.visualization is not None
        assert isinstance(result.visualization, str)
        
        # Validate corner coordinates
        for corner in result.corners:
            assert 0 <= corner[0] < image.shape[0]  # y-coordinate
            assert 0 <= corner[1] < image.shape[1]  # x-coordinate


def test_blob_detection(test_image):
    """Test blob detection."""
    image, _ = test_image
    
    result = detect_blobs(image, min_sigma=3.0, max_sigma=20.0, threshold=0.05, visualize=True)
    
    # Basic validation
    assert "blobs" in result.dict()
    assert len(result.blobs) > 0
    
    # Check visualization
    assert result.visualization is not None
    assert isinstance(result.visualization, str)
    
    # Validate blob properties
    for blob in result.blobs:
        y, x, sigma = blob
        assert 0 <= y < image.shape[0]
        assert 0 <= x < image.shape[1]
        assert sigma > 0


def test_daisy_features(test_image):
    """Test DAISY feature extraction."""
    image, _ = test_image
    
    result = detect_daisy(image, step=32, radius=15, visualize=True)
    
    # Basic validation
    assert "descriptors" in result.dict()
    assert "keypoints" in result.dict()
    assert len(result.descriptors) > 0
    assert len(result.keypoints) > 0
    
    # Check visualization
    assert result.visualization is not None
    assert isinstance(result.visualization, str)
    
    # Validate keypoint coordinates
    for keypoint in result.keypoints:
        y, x = keypoint
        assert 0 <= y < image.shape[0]
        assert 0 <= x < image.shape[1]


def test_hog_features(test_image):
    """Test HOG feature extraction."""
    image, _ = test_image
    
    result = detect_hog(image, orientations=8, pixels_per_cell=(16, 16), visualize=True)
    
    # Basic validation
    assert "features" in result.dict()
    assert len(result.features) > 0
    
    # Check visualization
    assert result.visualization is not None
    assert isinstance(result.visualization, str)


def test_haar_features(test_image):
    """Test Haar-like feature extraction."""
    image, _ = test_image
    
    for feature_type in ["basic", "extended"]:
        result = detect_haar(image, feature_type=feature_type, window_size=(32, 32), 
                           step_size=32, visualize=True)
        
        # Basic validation
        assert "features" in result.dict()
        assert "feature_locations" in result.dict()
        assert len(result.features) > 0
        assert len(result.feature_locations) > 0
        
        # Check visualization
        assert result.visualization is not None
        assert isinstance(result.visualization, str)
        
        # Validate feature locations
        for loc in result.feature_locations:
            y, x, h, w = loc
            assert 0 <= y < image.shape[0]
            assert 0 <= x < image.shape[1]
            assert h > 0
            assert w > 0


def test_template_matching(test_image, test_template):
    """Test template matching."""
    image, _ = test_image
    template, template_path = test_template
    
    for method in ["ssd", "ncc"]:
        result = detect_template(image, template, method=method, visualize=True)
        
        # Basic validation
        assert "matches" in result.dict()
        assert "scores" in result.dict()
        assert "best_match" in result.dict()
        assert "best_score" in result.dict()
        
        # Check matches
        assert len(result.matches) > 0
        assert len(result.scores) == len(result.matches)
        
        # Check visualization
        assert result.visualization is not None
        assert isinstance(result.visualization, str)
        
        # Validate match coordinates
        for match in result.matches:
            y, x = match
            assert 0 <= y < image.shape[0]
            assert 0 <= x < image.shape[1]


def test_lbp_features(test_image):
    """Test LBP feature extraction."""
    image, _ = test_image
    
    result = detect_lbp(image, radius=3, n_points=24, visualize=True)
    
    # Basic validation
    assert "histogram" in result.dict()
    assert len(result.histogram) > 0
    
    # Check visualization
    assert result.lbp_image is not None
    assert result.visualization is not None
    assert isinstance(result.visualization, str)


def test_peaks_detection(test_image):
    """Test peak detection."""
    image, _ = test_image
    
    result = detect_peaks(image, min_distance=10, threshold_abs=0.3, visualize=True)
    
    # Basic validation
    assert "peaks" in result.dict()
    assert "values" in result.dict()
    assert len(result.peaks) > 0
    assert len(result.values) == len(result.peaks)
    
    # Check visualization
    assert result.visualization is not None
    assert isinstance(result.visualization, str)
    
    # Validate peak coordinates
    for peak in result.peaks:
        y, x = peak
        assert 0 <= y < image.shape[0]
        assert 0 <= x < image.shape[1]


def test_censure_detection(test_image):
    """Test CENSURE feature detection."""
    image, _ = test_image
    
    result = detect_censure(image, min_scale=1, max_scale=7, visualize=True)
    
    # Basic validation
    assert "keypoints" in result.dict()
    assert len(result.keypoints) > 0
    
    # Check visualization
    assert result.visualization is not None
    assert isinstance(result.visualization, str)
    
    # Validate keypoint properties
    for keypoint in result.keypoints:
        assert hasattr(keypoint, 'x')
        assert hasattr(keypoint, 'y')
        assert hasattr(keypoint, 'size')
        assert 0 <= keypoint.y < image.shape[0]
        assert 0 <= keypoint.x < image.shape[1]
        assert keypoint.size > 0


def test_orb_features(test_image):
    """Test ORB feature extraction."""
    image, _ = test_image
    
    result = detect_orb(image, n_keypoints=100, visualize=True)
    
    # Basic validation
    assert "keypoints" in result.dict()
    assert "descriptors" in result.dict()
    assert len(result.keypoints) > 0
    assert len(result.descriptors) == len(result.keypoints)
    
    # Check visualization
    assert result.visualization is not None
    assert isinstance(result.visualization, str)
    
    # Validate keypoint properties
    for keypoint in result.keypoints:
        assert hasattr(keypoint, 'x')
        assert hasattr(keypoint, 'y')
        assert 0 <= keypoint.y < image.shape[0]
        assert 0 <= keypoint.x < image.shape[1]
    
    # Check binary descriptors
    for descriptor in result.descriptors:
        assert all(bit in [0, 1] for bit in descriptor)


def test_gabor_filter(test_image):
    """Test Gabor filtering."""
    image, _ = test_image
    
    result = detect_gabor(image, frequency=0.6, theta=0, visualize=True)
    
    # Basic validation
    assert "real" in result.dict()
    assert "imag" in result.dict()
    assert "magnitude" in result.dict()
    
    # Check visualizations
    assert result.real is not None
    assert result.imag is not None
    assert result.magnitude is not None
    assert result.visualization is not None
    assert isinstance(result.visualization, str)


def test_fisher_vectors(test_image):
    """Test Fisher vector encoding."""
    image, _ = test_image
    
    result = detect_fisher_vectors(image, n_components=2, random_state=42, visualize=True)
    
    # Basic validation
    assert "fisher_vector" in result.dict()
    assert "gmm_means" in result.dict()
    assert "gmm_covars" in result.dict()
    assert len(result.fisher_vector) > 0
    assert len(result.gmm_means) > 0
    assert len(result.gmm_covars) > 0
    
    # Check visualization
    assert result.visualization is not None
    assert isinstance(result.visualization, str)


def test_brief_features(test_image):
    """Test BRIEF feature extraction."""
    image, _ = test_image
    
    result = detect_brief(image, patch_size=49, visualize=True)
    
    # Basic validation
    assert "keypoints" in result.dict()
    assert "descriptors" in result.dict()
    assert len(result.keypoints) > 0
    assert len(result.descriptors) == len(result.keypoints)
    
    # Check visualization
    assert result.visualization is not None
    assert isinstance(result.visualization, str)
    
    # Validate keypoint properties
    for keypoint in result.keypoints:
        assert hasattr(keypoint, 'x')
        assert hasattr(keypoint, 'y')
        assert 0 <= keypoint.y < image.shape[0]
        assert 0 <= keypoint.x < image.shape[1]
    
    # Check binary descriptors
    for descriptor in result.descriptors:
        assert all(bit in [0, 1] for bit in descriptor)


def test_sift_features(test_image):
    """Test SIFT feature extraction."""
    image, _ = test_image
    
    result = detect_sift(image, n_octaves=4, threshold=0.05, visualize=True)
    
    # Basic validation
    assert "keypoints" in result.dict()
    assert "descriptors" in result.dict()
    assert len(result.keypoints) > 0
    assert len(result.descriptors) == len(result.keypoints)
    
    # Check visualization
    assert result.visualization is not None
    assert isinstance(result.visualization, str)
    
    # Validate keypoint properties
    for keypoint in result.keypoints:
        assert hasattr(keypoint, 'x')
        assert hasattr(keypoint, 'y')
        assert hasattr(keypoint, 'size')
        assert hasattr(keypoint, 'angle')
        assert 0 <= keypoint.y < image.shape[0]
        assert 0 <= keypoint.x < image.shape[1]
        assert keypoint.size > 0