#!/bin/bash
set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== mGBA-http Server Test Runner =====${NC}"

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

# Check if .NET project exists and build if necessary
if [ ! -d "mGBA-http/src/mGBAHttpServer" ]; then
    echo -e "${RED}mGBA-http project files not found. Check your installation.${NC}"
    exit 1
fi

# Always build the project to ensure compatibility
echo -e "${YELLOW}Building mGBA-http project...${NC}"
cd mGBA-http/src
dotnet build --configuration Release
cd ../..
echo -e "${GREEN}mGBA-http built successfully!${NC}"

# Start mGBA-http server using dotnet run (cross-platform execution)
echo -e "${YELLOW}Starting mGBA-http server...${NC}"
echo -e "${YELLOW}Note: .NET applications are cross-platform. The .dll file may show as a Windows format,${NC}"
echo -e "${YELLOW}      but the dotnet runtime handles platform-specific execution on macOS.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server when done testing.${NC}"
cd mGBA-http/src/mGBAHttpServer
dotnet run --no-build --configuration Release
cd ../../..

echo -e "${GREEN}===== Server testing completed =====${NC}" 