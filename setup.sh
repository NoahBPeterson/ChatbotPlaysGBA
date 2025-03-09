#!/bin/bash
set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== mGBA-http Python Utilities Setup =====${NC}"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Git is not installed. Please install git and try again.${NC}"
    exit 1
fi

# Check if dotnet is installed
if ! command -v dotnet &> /dev/null; then
    echo -e "${RED}dotnet is not installed. Please install .NET SDK and try again.${NC}"
    exit 1
fi

# Initialize git repo if not already initialized
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
    echo -e "${GREEN}Git repository initialized!${NC}"
else
    echo -e "${GREEN}Git repository already initialized.${NC}"
fi

# Handle the mGBA-http submodule
if [ -d "mGBA-http" ]; then
    echo -e "${YELLOW}Checking mGBA-http directory...${NC}"
    
    # Check if it's already a submodule
    if grep -q "mGBA-http" .gitmodules 2>/dev/null; then
        echo -e "${GREEN}mGBA-http is already set up as a submodule.${NC}"
    else
        # If it's a git repo but not a submodule
        if [ -d "mGBA-http/.git" ]; then
            echo -e "${YELLOW}mGBA-http is a git repository but not a submodule.${NC}"
            echo -e "${YELLOW}Removing existing mGBA-http and setting up as submodule...${NC}"
            rm -rf mGBA-http
            git submodule add https://github.com/nikouu/mGBA-http.git mGBA-http
            echo -e "${GREEN}mGBA-http added as a submodule!${NC}"
        else
            echo -e "${YELLOW}mGBA-http directory exists but is not a git repository.${NC}"
            echo -e "${YELLOW}Setting up mGBA-http as a submodule...${NC}"
            rm -rf mGBA-http
            git submodule add https://github.com/nikouu/mGBA-http.git mGBA-http
            echo -e "${GREEN}mGBA-http added as a submodule!${NC}"
        fi
    fi
else
    echo -e "${YELLOW}Setting up mGBA-http as a submodule...${NC}"
    git submodule add https://github.com/nikouu/mGBA-http.git mGBA-http
    echo -e "${GREEN}mGBA-http added as a submodule!${NC}"
fi

# Initialize and update submodules
echo -e "${YELLOW}Initializing and updating submodules...${NC}"
git submodule update --init --recursive
echo -e "${GREEN}Submodules initialized and updated!${NC}"

# Build mGBA-http with dotnet
echo -e "${YELLOW}Building mGBA-http with dotnet...${NC}"
cd mGBA-http
dotnet build
cd ..
echo -e "${GREEN}mGBA-http built successfully!${NC}"

# Set up Python virtual environment and install dependencies with uv
echo -e "${YELLOW}Setting up Python environment...${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv is not installed. Installing uv...${NC}"
    curl -sSf https://astral.sh/uv/install.sh | bash
    echo -e "${GREEN}uv installed!${NC}"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    uv venv
    echo -e "${GREEN}Virtual environment created!${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Install dependencies with uv
echo -e "${YELLOW}Installing dependencies with uv...${NC}"
uv pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed!${NC}"

# Make the run script executable
chmod +x run.sh

echo -e "${GREEN}===== Setup completed successfully! =====${NC}"
echo -e "${YELLOW}To run the mGBA-http server, use: ./run.sh${NC}" 