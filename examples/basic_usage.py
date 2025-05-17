"""
Basic usage examples for the Machine Vision MCP.

This script demonstrates how to use the Machine Vision MCP for various computer vision tasks.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io

from src.mcp import MachineLearningMCP


def save_example_images():
    """Save example images for testing."""
    # Create examples directory
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Save camera image
    camera = data.camera()
    io.imsave(examples_dir / "camera.png", camera)
    
    # Save astronaut image
    astronaut = data.astronaut()
    io.imsave(examples_dir / "astronaut.png", astronaut)
    
    # Save coins image
    coins = data.coins()
    io.imsave(examples_dir / "coins.png", coins)
    
    # Create a template from a part of the astronaut image
    template = astronaut[100:200, 150:250]
    io.imsave(examples_dir / "template.png", template)
    
    return examples_dir


def example_corner_detection(mcp, examples_dir):
    """Example of corner detection."""
    print("\n--- Corner Detection Example ---")
    
    # Detect corners using different methods
    methods = ["harris", "shi_tomasi", "kitchen_rosenfeld"]
    
    for method in methods:
        result = mcp.detect_corners(
            image_path=str(examples_dir / "camera.png"),
            method=method,
            visualize=True
        )
        
        print(f"{method.capitalize()} method: {len(result.corners)} corners detected")
    
    print("Corner detection completed.")


def example_blob_detection(mcp, examples_dir):
    """Example of blob detection."""
    print("\n--- Blob Detection Example ---")
    
    result = mcp.detect_blobs(
        image_path=str(examples_dir / "coins.png"),
        min_sigma=3.0,
        max_sigma=30.0,
        threshold=0.05,
        visualize=True
    )
    
    print(f"Detected {len(result.blobs)} blobs")
    print(f"Example blob (y, x, sigma): {result.blobs[0]}")
    
    print("Blob detection completed.")


def example_template_matching(mcp, examples_dir):
    """Example of template matching."""
    print("\n--- Template Matching Example ---")
    
    result = mcp.detect_template(
        image_path=str(examples_dir / "astronaut.png"),
        template_path=str(examples_dir / "template.png"),
        method="ncc",
        visualize=True
    )
    
    print(f"Found {len(result.matches)} matches")
    print(f"Best match at position (y, x): {result.best_match} with score: {result.best_score:.4f}")
    
    print("Template matching completed.")


def example_feature_extraction(mcp, examples_dir):
    """Example of various feature extraction methods."""
    print("\n--- Feature Extraction Examples ---")
    
    # HOG features
    hog_result = mcp.detect_hog(
        image_path=str(examples_dir / "camera.png"),
        orientations=8,
        pixels_per_cell=(16, 16),
        visualize=True
    )
    print(f"Extracted {len(hog_result.features)} HOG features")
    
    # DAISY features
    daisy_result = mcp.detect_daisy(
        image_path=str(examples_dir / "camera.png"),
        step=32,
        radius=15,
        visualize=True
    )
    print(f"Extracted {len(daisy_result.descriptors)} DAISY descriptors at {len(daisy_result.keypoints)} keypoints")
    
    # LBP features
    lbp_result = mcp.detect_lbp(
        image_path=str(examples_dir / "camera.png"),
        radius=3,
        n_points=24,
        visualize=True
    )
    print(f"Created LBP histogram with {len(lbp_result.histogram)} bins")
    
    # BRIEF features
    brief_result = mcp.detect_brief(
        image_path=str(examples_dir / "camera.png"),
        patch_size=49,
        visualize=True
    )
    print(f"Extracted BRIEF descriptors for {len(brief_result.keypoints)} keypoints")
    
    print("Feature extraction completed.")


def example_gabor_filtering(mcp, examples_dir):
    """Example of Gabor filtering."""
    print("\n--- Gabor Filtering Example ---")
    
    # Apply Gabor filter with different orientations
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for i, theta in enumerate(orientations):
        result = mcp.detect_gabor(
            image_path=str(examples_dir / "camera.png"),
            frequency=0.6,
            theta=theta,
            visualize=True
        )
        print(f"Applied Gabor filter with orientation {theta:.2f} radians")
    
    print("Gabor filtering completed.")


def example_fisher_vectors(mcp, examples_dir):
    """Example of Fisher vector encoding."""
    print("\n--- Fisher Vector Example ---")
    
    result = mcp.detect_fisher(
        image_path=str(examples_dir / "camera.png"),
        n_components=3,
        random_state=42,
        visualize=True
    )
    
    print(f"Created Fisher vector with {len(result.fisher_vector)} dimensions")
    print(f"Used a GMM with {len(result.gmm_means)} components")
    
    print("Fisher vector encoding completed.")


def main():
    """Main function demonstrating basic MCP usage."""
    print("Machine Vision MCP - Basic Usage Examples")
    print("=========================================")
    
    # Save example images
    examples_dir = save_example_images()
    print(f"Saved example images to {examples_dir}")
    
    # Initialize MCP
    mcp = MachineLearningMCP()
    
    # Test the decorator fix by running some examples
    print("\n--- Testing the decorator fix ---")
    try:
        # Test corner detection with explicit image_path parameter
        corner_result = mcp.detect_corners(
            image_path=str(examples_dir / "camera.png"),
            method="harris",
            visualize=True
        )
        print(f"Success! Detected {len(corner_result.corners)} corners using explicit image_path")
        
        # Run examples
        example_corner_detection(mcp, examples_dir)
        example_blob_detection(mcp, examples_dir)
        example_template_matching(mcp, examples_dir)
        example_feature_extraction(mcp, examples_dir)
        example_gabor_filtering(mcp, examples_dir)
        example_fisher_vectors(mcp, examples_dir)
        
        print("\nAll examples completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()