#!/bin/bash
set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== mGBA-http Server Runner =====${NC}"

# Check if mGBA is installed
if ! command -v mgba-qt &> /dev/null; then
    echo -e "${RED}mGBA is not installed or not in PATH.${NC}"
    echo -e "${YELLOW}Please install mGBA from https://mgba.io/downloads.html${NC}"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
    echo -e "${GREEN}Virtual environment activated!${NC}"
else
    echo -e "${RED}Virtual environment not found. Run ./setup.sh first.${NC}"
    exit 1
fi

# Check if dotnet is installed
if ! command -v dotnet &> /dev/null; then
    echo -e "${RED}dotnet is not installed. Please install .NET SDK and try again.${NC}"
    exit 1
fi

# Check if mGBA-http is built
if [ ! -f "mGBA-http/bin/Debug/net6.0/mGBA-http.dll" ]; then
    echo -e "${YELLOW}mGBA-http not built. Building now...${NC}"
    cd mGBA-http
    dotnet build
    cd ..
    echo -e "${GREEN}mGBA-http built successfully!${NC}"
fi

# Check if ROM file is provided
if [ "$#" -eq 1 ]; then
    ROM_FILE="$1"
    echo -e "${YELLOW}Using ROM file: ${ROM_FILE}${NC}"
    
    # Start mGBA with the ROM file
    echo -e "${YELLOW}Starting mGBA with ${ROM_FILE}...${NC}"
    mgba-qt "$ROM_FILE" &
    MGBA_PID=$!
    echo -e "${GREEN}mGBA started with PID: ${MGBA_PID}${NC}"
elif [ "$#" -gt 1 ]; then
    echo -e "${RED}Too many arguments.${NC}"
    echo -e "${YELLOW}Usage: ./run.sh [ROM_FILE]${NC}"
    exit 1
else
    echo -e "${YELLOW}No ROM file provided. Starting mGBA without a ROM...${NC}"
    mgba-qt &
    MGBA_PID=$!
    echo -e "${GREEN}mGBA started with PID: ${MGBA_PID}${NC}"
fi

# Wait for mGBA to initialize
echo -e "${YELLOW}Waiting for mGBA to initialize (3 seconds)...${NC}"
sleep 3

# Start mGBA-http server
echo -e "${YELLOW}Starting mGBA-http server...${NC}"
cd mGBA-http
dotnet run
cd ..

# Clean up
function cleanup {
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ -n "$MGBA_PID" ]; then
        echo -e "${YELLOW}Stopping mGBA (PID: ${MGBA_PID})...${NC}"
        kill $MGBA_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Set up trap to clean up on exit
trap cleanup EXIT

echo -e "${GREEN}===== Server started successfully! =====${NC}" 