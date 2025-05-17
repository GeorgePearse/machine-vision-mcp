"""
Pydantic schemas for the Machine Vision MCP.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field


class KeyPoint(BaseModel):
    """Representation of a keypoint detected in an image."""
    x: float
    y: float
    size: Optional[float] = None
    angle: Optional[float] = None
    response: Optional[float] = None
    octave: Optional[int] = None
    class_id: Optional[int] = None


class CornersResult(BaseModel):
    """Result of corner detection."""
    corners: List[Tuple[int, int]] = Field(..., description="List of corner coordinates (y, x)")
    scores: Optional[List[float]] = Field(None, description="Corner response scores")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class BlobsResult(BaseModel):
    """Result of blob detection."""
    blobs: List[Tuple[float, float, float]] = Field(..., 
                                               description="List of blobs as (y, x, sigma)")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class DaisyResult(BaseModel):
    """Result of DAISY feature extraction."""
    descriptors: List[List[float]] = Field(..., description="DAISY descriptors")
    keypoints: List[Tuple[int, int]] = Field(..., description="Keypoint coordinates (y, x)")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class HogResult(BaseModel):
    """Result of HOG feature extraction."""
    features: List[float] = Field(..., description="HOG feature vector")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class HaarResult(BaseModel):
    """Result of Haar-like feature extraction."""
    features: List[float] = Field(..., description="Haar feature values")
    feature_locations: List[Tuple[int, int, int, int]] = Field(..., 
                                                         description="Feature locations (y, x, h, w)")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class TemplateResult(BaseModel):
    """Result of template matching."""
    matches: List[Tuple[int, int]] = Field(..., description="Match positions (y, x)")
    scores: List[float] = Field(..., description="Match scores")
    best_match: Tuple[int, int] = Field(..., description="Best match position (y, x)")
    best_score: float = Field(..., description="Best match score")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class LbpResult(BaseModel):
    """Result of Local Binary Pattern extraction."""
    histogram: List[float] = Field(..., description="LBP histogram")
    lbp_image: Optional[str] = Field(None, description="Base64 encoded LBP image")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class PeaksResult(BaseModel):
    """Result of peak detection."""
    peaks: List[Tuple[int, int]] = Field(..., description="Peak positions (y, x)")
    values: List[float] = Field(..., description="Peak values")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class CensureResult(BaseModel):
    """Result of CENSURE feature detection."""
    keypoints: List[KeyPoint] = Field(..., description="CENSURE keypoints")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class OrbResult(BaseModel):
    """Result of ORB feature detection and description."""
    keypoints: List[KeyPoint] = Field(..., description="ORB keypoints")
    descriptors: List[List[int]] = Field(..., description="Binary descriptors")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class GaborResult(BaseModel):
    """Result of Gabor filtering."""
    real: Optional[str] = Field(None, description="Base64 encoded real part")
    imag: Optional[str] = Field(None, description="Base64 encoded imaginary part")
    magnitude: Optional[str] = Field(None, description="Base64 encoded magnitude")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class FisherResult(BaseModel):
    """Result of Fisher vector encoding."""
    fisher_vector: List[float] = Field(..., description="Fisher vector encoding")
    gmm_means: List[List[float]] = Field(..., description="GMM means")
    gmm_covars: List[List[float]] = Field(..., description="GMM covariances")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class BriefResult(BaseModel):
    """Result of BRIEF feature description."""
    keypoints: List[KeyPoint] = Field(..., description="Keypoints")
    descriptors: List[List[int]] = Field(..., description="Binary descriptors")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")


class SiftResult(BaseModel):
    """Result of SIFT feature detection and description."""
    keypoints: List[KeyPoint] = Field(..., description="SIFT keypoints")
    descriptors: List[List[float]] = Field(..., description="SIFT descriptors")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")