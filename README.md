# mGBA-http Python Utilities

This repository contains Python utilities for interfacing with mGBA emulator via its HTTP API.

## Overview

These tools allow you to control the mGBA emulator and access game memory through a simple API. They're particularly useful for tasks like:

- Programmatically controlling game inputs
- Reading and writing to memory
- Building automation tools and bots
- Developing AI agents that can play games
- Solving complex puzzles with computer vision

## Setup

### Prerequisites

- Python 3.6+
- [.NET SDK](https://dotnet.microsoft.com/download) (for building mGBA-http)
- [mGBA](https://mgba.io/downloads.html) installed and in your PATH
- git
- For vision features: Google Gemini API key with access to gemini-1.5-flash model

### Setting up the Gemini API Key

For vision-based features, you'll need a Google Gemini API key. You can set it up in two ways:

1. Create a `.env` file in the project root with the following content:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

2. Or pass it directly to scripts that require it with the `--api-key` parameter.

### Quick Setup

1. Clone this repository with submodules:
   ```bash
   git clone --recursive https://github.com/yourusername/mgba-http-python.git
   cd mgba-http-python
   ```

2. Run the setup script:
   ```bash
   ./setup.sh
   ```
   
   This script will:
   - Initialize the git repository if needed
   - Set up the mGBA-http submodule
   - Build the mGBA-http server with dotnet
   - Create a Python virtual environment with uv
   - Install dependencies

### Running the Server

To start the mGBA emulator and the HTTP server:

```bash
./run.sh [ROM_FILE]
```

Where `[ROM_FILE]` is the path to your ROM file (optional).

### Cross-Platform Notes

The mGBA-http server is built using .NET, which is cross-platform. When running on macOS, you might notice that the compiled DLL files appear to be in Windows PE format. This is normal for .NET assemblies, as the actual execution is handled by the platform-specific .NET runtime. The server will run properly on macOS, Windows, or Linux as long as the appropriate .NET runtime is installed.

## Components

### Base Components
- **MGBAController**: A high-level controller class for the mGBA-http API
- **emerald_party_reader.py**: A script for reading Pokémon party data from Pokémon Emerald
- **test_button_controls.py**: A test script for button control functionality
- **example_movement.py**: An example script that demonstrates character movement

### Vision Navigation Components
- **VisionController**: Enhanced controller with computer vision capabilities via Gemini
- **solve_ice_puzzle.py**: Example script for solving ice puzzles with vision guidance
- **VISION_NAVIGATION.md**: Documentation for the vision-based navigation system

## Memory Domains

The controller provides access to all GBA memory domains:

- **wram**: Main Work RAM (0x02000000-0x0203FFFF)
- **iwram**: Internal Work RAM (0x03000000-0x03007FFF)
- **bios**: GBA BIOS
- **cart0/1/2**: Cartridge ROM banks
- **sram**: Save RAM
- **vram**: Video RAM
- **oam**: Object Attribute Memory (sprites)
- **palette**: Color palettes
- **io**: I/O registers

## Usage Examples

### Basic Controller Usage

```python
from mgba_controller import MGBAController, Button

# Initialize the controller
controller = MGBAController()

# Press buttons
controller.press_button(Button.A)
controller.press_sequence([Button.UP, Button.A])

# Read memory
party_count = controller.read_byte(0x0244e9)  # For Pokemon Emerald
```

### Reading Pokémon Party Data

```bash
./emerald_party_reader.py
```

### Testing Button Controls

```bash
./test_button_controls.py
```

### Character Movement Example

```bash
./example_movement.py
```

### Vision-Based Navigation

```python
from vision_controller import VisionController
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Or set it directly
# api_key = "YOUR_GEMINI_API_KEY"

# Initialize with Gemini API key
vision_controller = VisionController(api_key=api_key)

# Navigate complex areas
vision_controller.navigate_complex_area("Pokemon Center")

# Solve ice puzzles
vision_controller.solve_ice_puzzle()
```

# Or use the complete script 
# With API key from .env file:
# ./solve_ice_puzzle.py
# Or with explicit API key:
# ./solve_ice_puzzle.py --api-key YOUR_GEMINI_API_KEY

See [VISION_NAVIGATION.md](VISION_NAVIGATION.md) for detailed information about the vision-based navigation system.

## Project Structure

- **mgba_controller.py**: Main controller class for interfacing with mGBA-http
- **vision_controller.py**: Enhanced controller with computer vision capabilities
- **mGBA-http/**: Submodule containing the C# HTTP server for mGBA
- **setup.sh**: Script for setting up the project
- **run.sh**: Script for running the mGBA emulator and HTTP server
- **emerald_party_reader.py**: Example script for reading Pokémon Emerald party data
- **test_button_controls.py**: Test script for button control functionality
- **example_movement.py**: Example script for character movement
- **solve_ice_puzzle.py**: Script for solving ice puzzles with computer vision

## Pokémon Game Support

Currently tested with:
- Pokémon Emerald

## License

MIT License 