#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}Starting Machine Vision MCP Application${NC}"
echo -e "${BLUE}===============================================${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
  echo -e "${YELLOW}Creating .env file...${NC}"
  cat > .env << EOL
# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key

# Server settings
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=5173
EOL
  echo -e "${GREEN}.env file created. Please edit it with your API keys.${NC}"
fi

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required dependencies
if ! command_exists python; then
  echo -e "${YELLOW}Python is not installed. Please install Python 3.8 or higher.${NC}"
  exit 1
fi

if ! command_exists node; then
  echo -e "${YELLOW}Node.js is not installed. Please install Node.js 14 or higher.${NC}"
  exit 1
fi

if ! command_exists npm; then
  echo -e "${YELLOW}npm is not installed. Please install npm.${NC}"
  exit 1
fi

# Create uploads directory if it doesn't exist
mkdir -p api/uploads

# Start the backend in the background
echo -e "${GREEN}Starting FastAPI backend...${NC}"
cd api
python -m pip install -r requirements.txt
cd ..
python -m uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Give the backend a moment to start
sleep 2
echo -e "${GREEN}Backend started at http://localhost:8000${NC}"

# Start the frontend in the background
echo -e "${GREEN}Installing frontend dependencies...${NC}"
cd frontend
npm install
echo -e "${GREEN}Starting React frontend...${NC}"
npm run dev &
FRONTEND_PID=$!

# Give the frontend a moment to start
sleep 2
echo -e "${GREEN}Frontend started at http://localhost:5173${NC}"

echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}Machine Vision MCP is now running!${NC}"
echo -e "${GREEN}API URL: http://localhost:8000${NC}"
echo -e "${GREEN}Frontend URL: http://localhost:5173${NC}"
echo -e "${GREEN}API Documentation: http://localhost:8000/docs${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Handle graceful shutdown
trap 'echo -e "${YELLOW}Stopping services...${NC}"; kill $BACKEND_PID; kill $FRONTEND_PID; echo -e "${GREEN}All services stopped.${NC}"; exit' INT

# Keep the script running
wait