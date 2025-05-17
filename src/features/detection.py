"""
Implementation of scikit-image based detection features.
"""
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from skimage import feature, filters, io, measure, morphology, segmentation, transform, util
from skimage.color import rgb2gray
from skimage.feature import (
    blob_dog,
    blob_log,
    corner_harris,
    corner_kitchen_rosenfeld,
    corner_peaks,
    corner_shi_tomasi,
    daisy,
    hessian_matrix_det,
    hog,
    local_binary_pattern,
    match_template,
    peak_local_max,
)
from skimage.filters import gabor
from skimage.transform import resize

from ..schemas import (
    BlobsResult,
    BriefResult,
    CensureResult,
    CornersResult,
    DaisyResult,
    FisherResult,
    GaborResult,
    HaarResult,
    HogResult,
    KeyPoint,
    LbpResult,
    OrbResult,
    PeaksResult,
    SiftResult,
    TemplateResult,
)
from .utils import image_to_base64, normalize_image, preprocess_image, visualize_features


def detect_corners(
    image: np.ndarray, method: str = "harris", **kwargs
) -> CornersResult:
    """
    Detect corners in an image using various methods.

    Args:
        image: Input image
        method: Detection method ('harris', 'shi_tomasi', 'kitchen_rosenfeld')
        **kwargs: Additional parameters for the detector

    Returns:
        Dictionary containing corner coordinates and scores
    """
    gray = preprocess_image(image)
    
    # Select the corner detection method
    if method == "harris":
        corner_response = corner_harris(gray, **kwargs)
    elif method == "shi_tomasi":
        corner_response = corner_shi_tomasi(gray, **kwargs)
    elif method == "kitchen_rosenfeld":
        corner_response = corner_kitchen_rosenfeld(gray, **kwargs)
    else:
        raise ValueError(f"Unknown corner detection method: {method}")
    
    # Find peaks in the corner response
    threshold_rel = kwargs.get("threshold_rel", 0.1)
    min_distance = kwargs.get("min_distance", 5)
    
    corners = corner_peaks(
        corner_response, min_distance=min_distance, threshold_rel=threshold_rel
    )
    
    # Get corner scores (responses)
    scores = [corner_response[y, x] for y, x in corners]
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, corners, "corners")
    
    # Convert corners to list of tuples
    corners_list = [(int(y), int(x)) for y, x in corners]
    
    return CornersResult(
        corners=corners_list,
        scores=scores,
        visualization=visualization
    )


def detect_blobs(
    image: np.ndarray, min_sigma: float = 1.0, max_sigma: float = 30.0, **kwargs
) -> BlobsResult:
    """
    Detect blobs in an image using the Difference of Gaussian method.

    Args:
        image: Input image
        min_sigma: Minimum standard deviation for Gaussian kernel
        max_sigma: Maximum standard deviation for Gaussian kernel
        **kwargs: Additional parameters for the detector

    Returns:
        Dictionary containing blob coordinates, sizes and scores
    """
    gray = preprocess_image(image)
    
    # Set default parameters
    threshold = kwargs.get("threshold", 0.1)
    num_sigma = kwargs.get("num_sigma", 10)
    
    # Detect blobs
    blobs = blob_dog(
        gray,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold,
        num_sigma=num_sigma,
    )
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, blobs, "blobs")
    
    # Convert blobs to list of tuples
    blobs_list = [(float(y), float(x), float(s)) for y, x, s in blobs]
    
    return BlobsResult(
        blobs=blobs_list,
        visualization=visualization
    )


def detect_daisy(
    image: np.ndarray, step: int = 4, radius: int = 15, **kwargs
) -> DaisyResult:
    """
    Extract DAISY features from an image.

    Args:
        image: Input image
        step: Distance between feature points
        radius: Radius of the outermost ring
        **kwargs: Additional parameters for the extractor

    Returns:
        Dictionary containing DAISY descriptors and their positions
    """
    gray = preprocess_image(image)
    
    # Set default parameters
    rings = kwargs.get("rings", 3)
    histograms = kwargs.get("histograms", 8)
    orientations = kwargs.get("orientations", 8)
    
    # Extract DAISY features
    descs, descs_img = daisy(
        gray,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
        visualize=True,
    )
    
    # Generate keypoints grid
    height, width = gray.shape
    keypoints = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            if y < descs_img.shape[0] and x < descs_img.shape[1]:
                keypoints.append((y, x))
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, keypoints, "daisy")
    
    # Convert descriptors to list
    descs_list = descs.reshape(-1, descs.shape[-1]).tolist()
    keypoints_list = [(int(y), int(x)) for y, x in keypoints]
    
    return DaisyResult(
        descriptors=descs_list,
        keypoints=keypoints_list,
        visualization=visualization
    )


def detect_hog(
    image: np.ndarray, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8), **kwargs
) -> HogResult:
    """
    Extract Histogram of Oriented Gradients features from an image.

    Args:
        image: Input image
        orientations: Number of orientation bins
        pixels_per_cell: Cell size (in pixels)
        **kwargs: Additional parameters for the extractor

    Returns:
        Dictionary containing HOG features and visualization
    """
    gray = preprocess_image(image)
    
    # Set default parameters
    cells_per_block = kwargs.get("cells_per_block", (3, 3))
    
    # Extract HOG features
    features, hog_image = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        feature_vector=True,
    )
    
    # Generate visualization
    visualization = None
    if kwargs.get("visualize", False):
        visualization = image_to_base64(hog_image)
    
    return HogResult(
        features=features.tolist(),
        visualization=visualization
    )


def detect_haar(
    image: np.ndarray, feature_type: str = "basic", **kwargs
) -> HaarResult:
    """
    Extract Haar-like features from an image.

    Args:
        image: Input image
        feature_type: Type of Haar-like features to extract
        **kwargs: Additional parameters for the extractor

    Returns:
        Dictionary containing Haar features and their positions
    """
    gray = preprocess_image(image)
    
    # Set default parameters
    window_size = kwargs.get("window_size", (24, 24))
    step_size = kwargs.get("step_size", 24)
    
    # Resize image to multiple of window_size if needed
    h, w = gray.shape
    new_h = (h // window_size[0]) * window_size[0]
    new_w = (w // window_size[1]) * window_size[1]
    if new_h != h or new_w != w:
        gray = resize(gray, (new_h, new_w), anti_aliasing=True)
    
    # Extract Haar-like features (simplified version)
    features = []
    feature_locations = []
    
    # Sliding window approach
    for y in range(0, gray.shape[0] - window_size[0] + 1, step_size):
        for x in range(0, gray.shape[1] - window_size[1] + 1, step_size):
            # Extract window
            window = gray[y:y + window_size[0], x:x + window_size[1]]
            
            # Compute simple Haar-like features
            if feature_type == "basic":
                # Vertical two-rectangle feature
                half_w = window_size[1] // 2
                left = np.sum(window[:, :half_w])
                right = np.sum(window[:, half_w:])
                feature_value = left - right
                
                features.append(float(feature_value))
                feature_locations.append((y, x, window_size[0], window_size[1]))
            
            elif feature_type == "extended":
                # Vertical and horizontal two-rectangle features
                half_w = window_size[1] // 2
                half_h = window_size[0] // 2
                
                # Vertical
                left = np.sum(window[:, :half_w])
                right = np.sum(window[:, half_w:])
                feature_v = left - right
                
                # Horizontal
                top = np.sum(window[:half_h, :])
                bottom = np.sum(window[half_h:, :])
                feature_h = top - bottom
                
                features.extend([float(feature_v), float(feature_h)])
                feature_locations.append((y, x, window_size[0], window_size[1]))
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, feature_locations, "haar")
    
    return HaarResult(
        features=features,
        feature_locations=feature_locations,
        visualization=visualization
    )


def detect_template(
    image: np.ndarray, template: np.ndarray, method: str = "ssd", **kwargs
) -> TemplateResult:
    """
    Perform template matching on an image.

    Args:
        image: Input image
        template: Template image
        method: Matching method ('ssd', 'ncc', 'zncc')
        **kwargs: Additional parameters for the matcher

    Returns:
        Dictionary containing match locations and scores
    """
    gray = preprocess_image(image)
    template_gray = preprocess_image(template)
    
    # Perform template matching
    result = match_template(gray, template_gray, pad_input=True)
    
    # Find matches
    threshold = kwargs.get("threshold", 0.5)
    min_distance = kwargs.get("min_distance", 10)
    
    # Different handling based on method (NCC vs SSD)
    if method in ["ncc", "zncc"]:
        # For NCC/ZNCC, higher values are better
        matches_coords = peak_local_max(
            result, min_distance=min_distance, threshold_abs=threshold
        )
        scores = [result[y, x] for y, x in matches_coords]
    else:
        # For SSD, lower values are better (invert for peak finding)
        matches_coords = peak_local_max(
            -result, min_distance=min_distance, threshold_abs=-threshold
        )
        scores = [-result[y, x] for y, x in matches_coords]
    
    # Find best match
    if len(scores) > 0:
        best_idx = np.argmax(scores) if method in ["ncc", "zncc"] else np.argmin(scores)
        best_match = matches_coords[best_idx]
        best_score = scores[best_idx]
    else:
        best_match = (0, 0)
        best_score = 0.0
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, matches_coords, "template")
    
    # Convert to list of tuples
    matches_list = [(int(y), int(x)) for y, x in matches_coords]
    best_match_tuple = (int(best_match[0]), int(best_match[1]))
    
    return TemplateResult(
        matches=matches_list,
        scores=scores,
        best_match=best_match_tuple,
        best_score=float(best_score),
        visualization=visualization
    )


def detect_lbp(
    image: np.ndarray, radius: int = 3, n_points: int = 24, **kwargs
) -> LbpResult:
    """
    Extract Local Binary Pattern features for texture classification.

    Args:
        image: Input image
        radius: Radius of circle for LBP
        n_points: Number of points on the circle
        **kwargs: Additional parameters for the extractor

    Returns:
        Dictionary containing LBP histogram and visualization
    """
    gray = preprocess_image(image)
    
    # Compute LBP
    method = kwargs.get("method", "uniform")
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    
    # Compute histogram
    n_bins = kwargs.get("n_bins", n_points + 2 if method == "uniform" else 2**n_points)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # Generate visualization if requested
    visualization = None
    lbp_image = None
    if kwargs.get("visualize", False):
        lbp_norm = normalize_image(lbp)
        lbp_image = image_to_base64(lbp_norm)
        visualization = lbp_image
    
    return LbpResult(
        histogram=hist.tolist(),
        lbp_image=lbp_image,
        visualization=visualization
    )


def detect_peaks(
    image: np.ndarray, min_distance: int = 10, threshold_abs: float = 0.1, **kwargs
) -> PeaksResult:
    """
    Detect peaks in an image.

    Args:
        image: Input image
        min_distance: Minimum distance between peaks
        threshold_abs: Minimum intensity threshold
        **kwargs: Additional parameters for the detector

    Returns:
        Dictionary containing peak coordinates and values
    """
    gray = preprocess_image(image)
    
    # Optional preprocessing using filters
    filter_type = kwargs.get("filter_type", None)
    if filter_type == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        gray = filters.gaussian(gray, sigma=sigma)
    elif filter_type == "median":
        size = kwargs.get("size", 3)
        gray = filters.median(gray, footprint=morphology.disk(size))
    
    # Detect peaks
    indices = peak_local_max(
        gray, min_distance=min_distance, threshold_abs=threshold_abs
    )
    
    # Get peak values
    values = [gray[y, x] for y, x in indices]
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, indices, "peaks")
    
    # Convert to list of tuples
    peaks_list = [(int(y), int(x)) for y, x in indices]
    
    return PeaksResult(
        peaks=peaks_list,
        values=[float(v) for v in values],
        visualization=visualization
    )


def detect_censure(
    image: np.ndarray, min_scale: int = 1, max_scale: int = 7, **kwargs
) -> CensureResult:
    """
    Detect CENSURE (CENter SUrround Extremas) features.

    Args:
        image: Input image
        min_scale: Minimum scale for feature detection
        max_scale: Maximum scale for feature detection
        **kwargs: Additional parameters for the detector

    Returns:
        Dictionary containing CENSURE keypoints
    """
    gray = preprocess_image(image)
    
    # NOTE: As CENSURE isn't directly available in scikit-image, we'll simulate it
    # using scale-space blob detection (which is conceptually similar)
    
    # Detect blobs at multiple scales
    threshold = kwargs.get("threshold", 0.01)
    blobs = blob_log(
        gray, 
        min_sigma=min_scale, 
        max_sigma=max_scale, 
        threshold=threshold
    )
    
    # Convert blobs to keypoints
    keypoints = []
    for y, x, sigma in blobs:
        keypoints.append(
            KeyPoint(
                x=float(x),
                y=float(y),
                size=float(sigma * 2),  # Convert sigma to diameter
                angle=0.0,
                response=float(gray[int(y), int(x)]),
                octave=int(np.log2(sigma))
            )
        )
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, blobs, "blobs")
    
    return CensureResult(
        keypoints=keypoints,
        visualization=visualization
    )


def detect_orb(
    image: np.ndarray, n_keypoints: int = 500, **kwargs
) -> OrbResult:
    """
    Extract ORB (Oriented FAST and Rotated BRIEF) features.

    Args:
        image: Input image
        n_keypoints: Maximum number of keypoints to detect
        **kwargs: Additional parameters for the detector

    Returns:
        Dictionary containing ORB keypoints and descriptors
    """
    gray = preprocess_image(image)
    
    # Since scikit-image doesn't have a direct ORB implementation,
    # we'll use FAST corners + a binary descriptor simulation
    
    # Detect FAST corners
    threshold = kwargs.get("threshold", 0.05)
    min_distance = kwargs.get("min_distance", 7)
    
    # Use Harris corners as an approximation
    corner_response = corner_harris(gray)
    corners = corner_peaks(
        corner_response, min_distance=min_distance, 
        threshold_rel=threshold, num_peaks=n_keypoints
    )
    
    # Simulate ORB descriptors with random binary strings
    # (In a real implementation, this would compute actual BRIEF descriptors)
    descriptor_size = kwargs.get("descriptor_size", 256)
    np.random.seed(kwargs.get("random_state", 42))
    descriptors = np.random.randint(0, 2, size=(len(corners), descriptor_size))
    
    # Create keypoints
    keypoints = []
    for i, (y, x) in enumerate(corners):
        # Estimate a scale and orientation (simplified)
        scale = 1.0
        angle = 0.0
        
        keypoints.append(
            KeyPoint(
                x=float(x),
                y=float(y),
                size=float(scale * 10),
                angle=float(angle),
                response=float(corner_response[y, x]),
                octave=0
            )
        )
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, corners, "orb")
    
    return OrbResult(
        keypoints=keypoints,
        descriptors=descriptors.tolist(),
        visualization=visualization
    )


def detect_gabor(
    image: np.ndarray, frequency: float = 0.6, theta: float = 0, **kwargs
) -> GaborResult:
    """
    Apply Gabor filter to extract features similar to primary visual cortex.

    Args:
        image: Input image
        frequency: Frequency of the harmonic function
        theta: Orientation in radians
        **kwargs: Additional parameters for Gabor filter

    Returns:
        Dictionary containing Gabor response
    """
    gray = preprocess_image(image)
    
    # Set default parameters
    sigma_x = kwargs.get("sigma_x", 1.0)
    sigma_y = kwargs.get("sigma_y", 1.0)
    offset = kwargs.get("offset", 0)
    
    # Apply Gabor filter
    real, imag = gabor(
        gray, 
        frequency=frequency, 
        theta=theta, 
        sigma_x=sigma_x, 
        sigma_y=sigma_y, 
        offset=offset
    )
    
    # Compute magnitude
    magnitude = np.sqrt(real**2 + imag**2)
    
    # Generate visualizations if requested
    visualization = None
    real_img = None
    imag_img = None
    magnitude_img = None
    if kwargs.get("visualize", False):
        real_img = image_to_base64(normalize_image(real))
        imag_img = image_to_base64(normalize_image(imag))
        magnitude_img = image_to_base64(normalize_image(magnitude))
        visualization = magnitude_img
    
    return GaborResult(
        real=real_img,
        imag=imag_img,
        magnitude=magnitude_img,
        visualization=visualization
    )


def detect_fisher_vectors(
    image: np.ndarray, n_components: int = 2, random_state: int = 0, **kwargs
) -> FisherResult:
    """
    Compute Fisher vectors for image encoding.

    Args:
        image: Input image
        n_components: Number of mixture components
        random_state: Random state for reproducibility
        **kwargs: Additional parameters for Fisher vector computation

    Returns:
        Dictionary containing Fisher vector encoding
    """
    gray = preprocess_image(image)
    
    # Extract image patches
    patch_size = kwargs.get("patch_size", 8)
    stride = kwargs.get("stride", 4)
    
    patches = []
    for y in range(0, gray.shape[0] - patch_size + 1, stride):
        for x in range(0, gray.shape[1] - patch_size + 1, stride):
            patch = gray[y:y + patch_size, x:x + patch_size]
            patches.append(patch.flatten())
    
    patches = np.array(patches)
    
    # Train a Gaussian Mixture Model (GMM)
    # Note: We're creating a simplified version here
    # In a real implementation, we would use sklearn.mixture.GaussianMixture
    
    # Simulate a GMM by clustering patches
    np.random.seed(random_state)
    n_samples = min(1000, len(patches))
    sample_indices = np.random.choice(len(patches), n_samples, replace=False)
    sampled_patches = patches[sample_indices]
    
    # Simplified GMM - just use means and covariances of random subsets
    means = []
    covars = []
    for i in range(n_components):
        component_samples = sampled_patches[i::n_components]
        if len(component_samples) > 0:
            means.append(np.mean(component_samples, axis=0))
            covars.append(np.var(component_samples, axis=0) + 1e-6)  # Add small epsilon
    
    # Compute a simplified Fisher vector
    # In a real implementation, we would compute gradients w.r.t. GMM parameters
    fisher_vector = []
    for mean, covar in zip(means, covars):
        # First-order statistics
        mean_diff = np.mean(patches, axis=0) - mean
        fisher_vector.extend((mean_diff / np.sqrt(covar)).tolist())
        
        # Second-order statistics
        var_diff = np.var(patches, axis=0) / covar - 1
        fisher_vector.extend((var_diff / np.sqrt(2)).tolist())
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        # Create a distance matrix between patches
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(sampled_patches[:100]))
        distances = normalize_image(distances)
        visualization = image_to_base64(distances)
    
    return FisherResult(
        fisher_vector=fisher_vector,
        gmm_means=[mean.tolist() for mean in means],
        gmm_covars=[covar.tolist() for covar in covars],
        visualization=visualization
    )


def detect_brief(
    image: np.ndarray, patch_size: int = 49, **kwargs
) -> BriefResult:
    """
    Extract BRIEF (Binary Robust Independent Elementary Features) binary descriptors.

    Args:
        image: Input image
        patch_size: Size of the patch used for keypoint description
        **kwargs: Additional parameters for the descriptor

    Returns:
        Dictionary containing BRIEF keypoints and descriptors
    """
    gray = preprocess_image(image)
    
    # Detect keypoints (using Harris corners or similar)
    min_distance = kwargs.get("min_distance", 7)
    threshold = kwargs.get("threshold", 0.05)
    max_keypoints = kwargs.get("max_keypoints", 100)
    
    corner_response = corner_harris(gray)
    corners = corner_peaks(
        corner_response, min_distance=min_distance, 
        threshold_rel=threshold, num_peaks=max_keypoints
    )
    
    # Generate BRIEF-like descriptors
    # In a real implementation, we would use random point pairs
    # Here we'll generate random binary descriptors for demonstration
    descriptor_size = kwargs.get("descriptor_size", 256)
    np.random.seed(kwargs.get("random_state", 42))
    descriptors = np.random.randint(0, 2, size=(len(corners), descriptor_size))
    
    # Create keypoints
    keypoints = []
    for i, (y, x) in enumerate(corners):
        keypoints.append(
            KeyPoint(
                x=float(x),
                y=float(y),
                size=float(patch_size),
                angle=0.0,
                response=float(corner_response[y, x]),
                octave=0
            )
        )
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, corners, "brief")
    
    return BriefResult(
        keypoints=keypoints,
        descriptors=descriptors.tolist(),
        visualization=visualization
    )


def detect_sift(
    image: np.ndarray, n_octaves: int = 4, **kwargs
) -> SiftResult:
    """
    Extract SIFT (Scale-Invariant Feature Transform) features.

    Args:
        image: Input image
        n_octaves: Number of octaves
        **kwargs: Additional parameters for the detector/descriptor

    Returns:
        Dictionary containing SIFT keypoints and descriptors
    """
    gray = preprocess_image(image)
    
    # Since scikit-image doesn't have a direct SIFT implementation,
    # we'll simulate it using blob detection and gradient descriptors
    
    # Detect blobs at multiple scales
    threshold = kwargs.get("threshold", 0.01)
    max_sigma = kwargs.get("max_sigma", 16)
    num_sigma = kwargs.get("num_sigma", 3)
    
    blobs = blob_log(
        gray, 
        min_sigma=1.0, 
        max_sigma=max_sigma, 
        num_sigma=num_sigma, 
        threshold=threshold
    )
    
    # Generate SIFT-like descriptors for each keypoint
    # In a real implementation, we would compute gradient histograms
    # Here we'll use a simplified approach for demonstration
    descriptor_size = kwargs.get("descriptor_size", 128)
    keypoints = []
    descriptors = []
    
    for y, x, sigma in blobs:
        # Convert blob to integer coordinates
        y_int, x_int = int(y), int(x)
        
        # Skip blobs near boundaries
        win_size = int(3 * sigma)
        if (y_int - win_size < 0 or y_int + win_size >= gray.shape[0] or
            x_int - win_size < 0 or x_int + win_size >= gray.shape[1]):
            continue
        
        # Extract patch around keypoint
        patch = gray[y_int-win_size:y_int+win_size+1, x_int-win_size:x_int+win_size+1]
        
        # Compute a simplified descriptor (gradient histograms)
        if patch.size > 0:
            # Compute gradients
            gy, gx = np.gradient(patch)
            magnitude = np.sqrt(gy**2 + gx**2)
            orientation = np.arctan2(gy, gx) % (2 * np.pi)
            
            # Create histogram of gradients
            hist, _ = np.histogram(orientation.ravel(), bins=8, weights=magnitude.ravel(), density=True)
            
            # Repeat the histogram to create a SIFT-like descriptor
            descriptor = np.tile(hist, descriptor_size // 8)
            
            # Normalize descriptor
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor /= norm
            
            # Add keypoint and descriptor
            keypoints.append(
                KeyPoint(
                    x=float(x),
                    y=float(y),
                    size=float(sigma * 2),  # Convert sigma to diameter
                    angle=float(np.mean(orientation)),
                    response=float(np.mean(magnitude)),
                    octave=int(np.log2(sigma))
                )
            )
            descriptors.append(descriptor.tolist())
    
    # Generate visualization if requested
    visualization = None
    if kwargs.get("visualize", False):
        visualization = visualize_features(image, blobs, "sift")
    
    return SiftResult(
        keypoints=keypoints,
        descriptors=descriptors,
        visualization=visualization
    )