"""
Machine Vision MCP - A Fast-MCP implementation for scikit-image detection features.
"""
import functools
from typing import Dict, List, Optional, Union, Callable, Any

import numpy as np
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from skimage.io import imread

from .features.detection import (
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
from .schemas import (
    BlobsResult,
    BriefResult,
    CensureResult,
    CornersResult,
    DaisyResult,
    FisherResult,
    GaborResult,
    HaarResult,
    HogResult,
    LbpResult,
    OrbResult,
    PeaksResult,
    SiftResult,
    TemplateResult,
)


def image_path_to_image(func: Callable) -> Callable:
    """
    Decorator to convert image_path parameter to image array before passing to detection functions.
    This fixes the issue with FastMCP passing image_path directly to the detection functions.
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function that handles image_path conversion
    """
    @functools.wraps(func)
    def wrapper(self, image_path: str, *args, **kwargs):
        # Remove image_path from kwargs if it's there (prevents it from being passed twice)
        if 'image_path' in kwargs:
            del kwargs['image_path']
        
        # Load the image
        image = imread(image_path)
        
        # Call the original function with the image and other args
        return func(self, image, *args, **kwargs)
    
    return wrapper


class MachineLearningMCP(FastMCP):
    """MCP for computer vision operations using scikit-image."""

    @image_path_to_image
    @FastMCP.tool
    def detect_corners(
        self, image: np.ndarray, method: str = "harris", **kwargs
    ) -> CornersResult:
        """
        Detect corners in an image using various methods.

        Args:
            image_path: Path to the input image
            method: Detection method ('harris', 'shi_tomasi', 'kitchen_rosenfeld', etc.)
            **kwargs: Additional parameters for the detector

        Returns:
            Dictionary containing corner coordinates and scores
        """
        return detect_corners(image, method=method, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_blobs(
        self, image: np.ndarray, min_sigma: float = 1.0, max_sigma: float = 30.0, **kwargs
    ) -> BlobsResult:
        """
        Detect blobs in an image using the Difference of Gaussian method.

        Args:
            image_path: Path to the input image
            min_sigma: Minimum standard deviation for Gaussian kernel
            max_sigma: Maximum standard deviation for Gaussian kernel
            **kwargs: Additional parameters for the detector

        Returns:
            Dictionary containing blob coordinates, sizes and scores
        """
        return detect_blobs(image, min_sigma=min_sigma, max_sigma=max_sigma, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_daisy(
        self, image: np.ndarray, step: int = 4, radius: int = 15, **kwargs
    ) -> DaisyResult:
        """
        Extract DAISY features from an image.

        Args:
            image_path: Path to the input image
            step: Distance between feature points
            radius: Radius of the outermost ring
            **kwargs: Additional parameters for the extractor

        Returns:
            Dictionary containing DAISY descriptors and their positions
        """
        return detect_daisy(image, step=step, radius=radius, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_hog(
        self, image: np.ndarray, orientations: int = 9, pixels_per_cell: tuple = (8, 8), **kwargs
    ) -> HogResult:
        """
        Extract Histogram of Oriented Gradients features from an image.

        Args:
            image_path: Path to the input image
            orientations: Number of orientation bins
            pixels_per_cell: Cell size (in pixels)
            **kwargs: Additional parameters for the extractor

        Returns:
            Dictionary containing HOG features and visualization
        """
        return detect_hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_haar(
        self, image: np.ndarray, feature_type: str = "basic", **kwargs
    ) -> HaarResult:
        """
        Extract Haar-like features from an image.

        Args:
            image_path: Path to the input image
            feature_type: Type of Haar-like features to extract
            **kwargs: Additional parameters for the extractor

        Returns:
            Dictionary containing Haar features and their positions
        """
        return detect_haar(image, feature_type=feature_type, **kwargs)

    @FastMCP.tool
    def detect_template(
        self, image_path: str, template_path: str, method: str = "ssd", **kwargs
    ) -> TemplateResult:
        """
        Perform template matching on an image.

        Args:
            image_path: Path to the input image
            template_path: Path to the template image
            method: Matching method ('ssd', 'ncc', 'zncc')
            **kwargs: Additional parameters for the matcher

        Returns:
            Dictionary containing match locations and scores
        """
        # Special case for template matching as it needs two images
        if 'image_path' in kwargs:
            del kwargs['image_path']
        image = imread(image_path)
        template = imread(template_path)
        return detect_template(image, template, method=method, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_lbp(
        self, image: np.ndarray, radius: int = 3, n_points: int = 24, **kwargs
    ) -> LbpResult:
        """
        Extract Local Binary Pattern features for texture classification.

        Args:
            image_path: Path to the input image
            radius: Radius of circle for LBP
            n_points: Number of points on the circle
            **kwargs: Additional parameters for the extractor

        Returns:
            Dictionary containing LBP histogram and visualization
        """
        return detect_lbp(image, radius=radius, n_points=n_points, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_peaks(
        self, image: np.ndarray, min_distance: int = 10, threshold_abs: float = 0.1, **kwargs
    ) -> PeaksResult:
        """
        Detect peaks in an image.

        Args:
            image_path: Path to the input image
            min_distance: Minimum distance between peaks
            threshold_abs: Minimum intensity threshold
            **kwargs: Additional parameters for the detector

        Returns:
            Dictionary containing peak coordinates and values
        """
        return detect_peaks(image, min_distance=min_distance, threshold_abs=threshold_abs, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_censure(
        self, image: np.ndarray, min_scale: int = 1, max_scale: int = 7, **kwargs
    ) -> CensureResult:
        """
        Detect CENSURE (CENter SUrround Extremas) features.

        Args:
            image_path: Path to the input image
            min_scale: Minimum scale for feature detection
            max_scale: Maximum scale for feature detection
            **kwargs: Additional parameters for the detector

        Returns:
            Dictionary containing CENSURE keypoints
        """
        return detect_censure(image, min_scale=min_scale, max_scale=max_scale, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_orb(
        self, image: np.ndarray, n_keypoints: int = 500, **kwargs
    ) -> OrbResult:
        """
        Extract ORB (Oriented FAST and Rotated BRIEF) features.

        Args:
            image_path: Path to the input image
            n_keypoints: Maximum number of keypoints to detect
            **kwargs: Additional parameters for the detector

        Returns:
            Dictionary containing ORB keypoints and descriptors
        """
        return detect_orb(image, n_keypoints=n_keypoints, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_gabor(
        self, image: np.ndarray, frequency: float = 0.6, theta: float = 0, **kwargs
    ) -> GaborResult:
        """
        Apply Gabor filter to extract features similar to primary visual cortex.

        Args:
            image_path: Path to the input image
            frequency: Frequency of the harmonic function
            theta: Orientation in radians
            **kwargs: Additional parameters for Gabor filter

        Returns:
            Dictionary containing Gabor response
        """
        return detect_gabor(image, frequency=frequency, theta=theta, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_fisher(
        self, image: np.ndarray, n_components: int = 2, random_state: int = 0, **kwargs
    ) -> FisherResult:
        """
        Compute Fisher vectors for image encoding.

        Args:
            image_path: Path to the input image
            n_components: Number of mixture components
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for Fisher vector computation

        Returns:
            Dictionary containing Fisher vector encoding
        """
        return detect_fisher_vectors(image, n_components=n_components, random_state=random_state, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_brief(
        self, image: np.ndarray, patch_size: int = 49, **kwargs
    ) -> BriefResult:
        """
        Extract BRIEF (Binary Robust Independent Elementary Features) binary descriptors.

        Args:
            image_path: Path to the input image
            patch_size: Size of the patch used for keypoint description
            **kwargs: Additional parameters for the descriptor

        Returns:
            Dictionary containing BRIEF keypoints and descriptors
        """
        return detect_brief(image, patch_size=patch_size, **kwargs)

    @image_path_to_image
    @FastMCP.tool
    def detect_sift(
        self, image: np.ndarray, n_octaves: int = 4, **kwargs
    ) -> SiftResult:
        """
        Extract SIFT (Scale-Invariant Feature Transform) features.

        Args:
            image_path: Path to the input image
            n_octaves: Number of octaves
            **kwargs: Additional parameters for the detector/descriptor

        Returns:
            Dictionary containing SIFT keypoints and descriptors
        """
        return detect_sift(image, n_octaves=n_octaves, **kwargs)


def register_openai_tools():
    """
    Register the MCP tools with OpenAI's tool format.

    Returns:
        List of tool definitions compatible with OpenAI's API
    """
    mcp = MachineLearningMCP()
    return mcp.to_openai_tools()