# Machine Vision MCP

A Machine Control Protocol (MCP) for connecting LLMs to scikit-image tools, built with [fast-mcp](https://github.com/jlowin/fastmcp). This project enables language models to perform complex computer vision tasks through a clean, accessible API.

## Features

This MCP focuses on the detection capabilities from scikit-image, including:

- Dense DAISY feature description
- Histogram of Oriented Gradients (HOG)
- Haar-like feature descriptors
- Template matching
- Corner detection
- Multi-Block Local Binary Pattern for texture classification
- Filling holes and finding peaks
- CENSURE feature detector
- Blob detection
- ORB feature detector and binary descriptor
- Gabors/Primary Visual Cortex "Simple Cells"
- Fisher vector feature encoding
- BRIEF binary descriptor
- SIFT feature detector and descriptor

These tools enable LLMs to:
- Detect and measure objects in images
- Extract meaningful features for classification
- Perform advanced image analysis with minimal coding
- Automate complex computer vision workflows

## Architecture

The project is structured as follows:

```
machine-vision-mcp/
├── api/                  # FastAPI backend
│   ├── app/              # API application
│   │   ├── routers/      # API endpoints
│   │   ├── main.py       # Entry point
│   │   └── claude_integration.py  # Claude API integration
│   ├── requirements.txt  # Backend dependencies
│   └── Dockerfile        # Backend container definition
├── frontend/             # React + Vite frontend
│   ├── src/              # React components and pages
│   ├── package.json      # Frontend dependencies
│   └── Dockerfile        # Frontend container definition
├── src/                  # MCP implementation
│   ├── mcp.py            # Main MCP definition
│   ├── features/         # Feature detection implementations
│   └── schemas/          # Data models
├── tests/                # Test suite
├── examples/             # Example usage
├── docker-compose.yml    # Docker Compose configuration
└── start.sh              # Startup script for local development
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- Docker and Docker Compose (optional)
- Anthropic API key for Claude integration (optional)
- OpenAI API key for GPT-4 Vision integration (optional)

### Installation and Setup

#### Option 1: Using the startup script (local development)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/machine-vision-mcp.git
   cd machine-vision-mcp
   ```

2. Run the startup script:
   ```bash
   ./start.sh
   ```

   This script will:
   - Create a `.env` file (if not present)
   - Install backend dependencies
   - Install frontend dependencies
   - Start the FastAPI backend and React frontend
   - Provide URLs for accessing the services

3. Access the application:
   - Frontend: http://localhost:5173
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

#### Option 2: Using Docker Compose

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/machine-vision-mcp.git
   cd machine-vision-mcp
   ```

2. Copy the environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Start the application with Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - Frontend: http://localhost:5173
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Usage

### Web Interface

The web interface provides:

1. **Image Upload**: Upload images for analysis
2. **Analysis Tools**: Choose from various scikit-image tools:
   - Corner Detection
   - Blob Detection
   - Histogram of Oriented Gradients (HOG)
   - DAISY Features
   - Local Binary Patterns
   - Object Segmentation
   - Pixel Count
   - Area Measurement
3. **Claude/GPT-4 Integration**: Ask Claude or GPT-4 to analyze images with natural language
4. **Result Visualization**: View detailed analysis results with visual overlays

### API Usage

The API provides endpoints for:

1. **Image Upload**: `/api/upload`
2. **Image Analysis**: `/api/analyze`
3. **Claude Analysis**: `/api/claude/analyze`
4. **OpenAI Analysis**: `/api/openai/analyze`

Example API request using cURL:

```bash
# Upload an image
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/api/upload

# Analyze the image using scikit-image
curl -X POST -H "Content-Type: application/json" \
  -d '{"filename":"uploaded_image.jpg","analysis_type":"corner_detection"}' \
  http://localhost:8000/api/analyze

# Analyze with Claude
curl -X POST -H "Content-Type: application/json" \
  -d '{"filename":"uploaded_image.jpg","prompt":"Count the objects in this image"}' \
  http://localhost:8000/api/claude/analyze
```

### Programmatic Usage

```python
from machine_vision_mcp import MachineLearningMCP

# Initialize the MCP
mcp = MachineLearningMCP()

# Use a detection feature
result = mcp.detect_corners(image_path="path/to/image.jpg")
print(f"Found {len(result.corners)} corners")

# Use blob detection
blobs = mcp.detect_blobs(image_path="path/to/image.jpg")
print(f"Found {len(blobs.blobs)} blobs")
```

## LLM Integration

This project includes integrations with:

1. **Claude API**: Uses Claude-3-Opus for high-quality image understanding
2. **OpenAI API**: Includes GPT-4 Vision as an alternative option

To use these features, you'll need to:

1. Add your API keys to the `.env` file
2. Use the Claude/OpenAI tabs in the frontend interface
3. Or call the respective API endpoints

## Development

### Running Tests

```bash
# From project root
pytest tests/
```

### Adding New Features

1. Add implementation in `src/features/`
2. Update schemas in `src/schemas/`
3. Expose in `src/mcp.py`
4. Add to API in `api/app/routers/analysis.py`
5. Add to frontend in `frontend/src/pages/AnalysisPage.jsx`

## Future Enhancements

Planned features for upcoming releases:

- Integration with additional scikit-image modules
- Support for video analysis
- More advanced object classification
- Auto-generation of Python code from LLM descriptions
- Batch processing for multiple images
- Enhanced visualization options
- Performance optimizations for larger images

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.