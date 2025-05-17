"""
Example usage of Machine Vision MCP with LLMs.

This script demonstrates how to use the Machine Vision MCP with different LLM providers.
"""
import base64
import json
import os
from typing import Dict, List

from src.mcp import MachineLearningMCP, register_openai_tools


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for including in LLM messages."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Example 1: Using OpenAI
def example_openai():
    """Example using OpenAI's API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI Python package not installed. Run: pip install openai")
        return
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Register MCP tools with OpenAI
    tools = register_openai_tools()
    
    # Example image path - replace with your image
    image_path = "examples/sample_image.jpg"
    image_base64 = encode_image_to_base64(image_path)
    
    # Create a message with the image
    messages = [
        {"role": "system", "content": "You are a computer vision assistant. Use the available tools to analyze images."},
        {"role": "user", "content": [
            {"type": "text", "text": "Count the number of distinct objects in this image and describe what they are."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}
    ]
    
    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",  # Make sure to use a model that supports vision
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    # Process and print the response
    print("OpenAI Response:")
    print(response.choices[0].message.content)
    
    # If tool calls were made, handle them
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        # Handle tool calls
        mcp = MachineLearningMCP()
        tool_calls = response.choices[0].message.tool_calls
        
        for tool_call in tool_calls:
            # Extract tool and parameters
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print(f"\nTool Call: {function_name}")
            print(f"Arguments: {arguments}")
            
            # Execute the tool (in a real application, you would call the actual tool)
            if hasattr(mcp, function_name):
                try:
                    # Replace image paths with actual image path
                    if "image_path" in arguments:
                        arguments["image_path"] = image_path
                    if "template_path" in arguments and "template" in arguments["template_path"]:
                        arguments["template_path"] = "examples/template.jpg"
                    
                    # Call the MCP function
                    result = getattr(mcp, function_name)(**arguments)
                    print(f"Tool Result: {result}\n")
                except Exception as e:
                    print(f"Error executing tool: {e}")


# Example 2: Using Claude (Anthropic)
def example_anthropic():
    """Example using Anthropic's Claude API."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Anthropic Python package not installed. Run: pip install anthropic")
        return
    
    # Initialize Anthropic client
    client = Anthropic()
    
    # Example image path - replace with your image
    image_path = "examples/sample_image.jpg"
    image_base64 = encode_image_to_base64(image_path)
    
    # Create a message with the image
    message = client.messages.create(
        model="claude-3-opus-20240229",  # Make sure to use a model that supports vision
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image. I want to detect all corners and key points in it."},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}}
                ]
            }
        ]
    )
    
    # Print the response
    print("Claude Response:")
    print(message.content)
    
    # Note: Claude doesn't have native function calling like OpenAI
    # We would need to parse the instructions from Claude and then call our MCP manually
    print("\nManually calling MCP based on Claude's analysis...")
    
    mcp = MachineLearningMCP()
    corners_result = mcp.detect_corners(image_path=image_path, visualize=True)
    
    print(f"Detected {len(corners_result.corners)} corners in the image")


# Example 3: Manual MCP usage with any LLM
def example_manual_mcp():
    """Example of manually using the MCP with any LLM response."""
    # Initialize MCP
    mcp = MachineLearningMCP()
    
    # Example image path
    image_path = "examples/sample_image.jpg"
    
    print("Manual MCP Usage:")
    
    # Example 1: Corner detection
    print("\nRunning corner detection...")
    corners_result = mcp.detect_corners(image_path=image_path, method="harris", visualize=True)
    print(f"Detected {len(corners_result.corners)} corners")
    
    # Example 2: Blob detection
    print("\nRunning blob detection...")
    blobs_result = mcp.detect_blobs(image_path=image_path, min_sigma=3.0, max_sigma=30.0, visualize=True)
    print(f"Detected {len(blobs_result.blobs)} blobs")
    
    # Example 3: HOG features
    print("\nExtracting HOG features...")
    hog_result = mcp.detect_hog(image_path=image_path, visualize=True)
    print(f"Extracted {len(hog_result.features)} HOG features")
    
    # Example 4: LBP for texture analysis
    print("\nExtracting LBP features for texture analysis...")
    lbp_result = mcp.detect_lbp(image_path=image_path, visualize=True)
    print(f"Created LBP histogram with {len(lbp_result.histogram)} bins")
    
    # This data could then be integrated into any LLM system


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Run examples
    print("Running OpenAI example:")
    print("-" * 50)
    example_openai()
    
    print("\n\nRunning Claude example:")
    print("-" * 50)
    example_anthropic()
    
    print("\n\nRunning Manual MCP example:")
    print("-" * 50)
    example_manual_mcp()