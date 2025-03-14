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

There's also support for OpenAI, Deepseek, and Claude.

2. Or pass it directly to scripts that require it with the `--api-key` parameter.

### Quick Setup

1. Clone this repository with submodules:
   ```bash
   git clone --recursive https://github.com/NoahBPeterson/ChatbotPlaysGBA.git
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

3. Install dependencies

 * Install mgba as a command-line utility (macOS & Linux)
      ```
      uv venv
      source .venv/bin/activate
      uv pip install -r requirements.txt
      ```
 * This emulator does not work without a Gameboy Advance ROM.

### Running the Server

To start the mGBA emulator and the HTTP server:

```bash
./run.sh [ROM_FILE]
```

Where `[ROM_FILE]` is the path to your ROM file (optional).

If you do not have mgba installed on PATH, you will have to run it yourself and load the ROM.

You must add the `mGBASocketServer.lua` script from the `/mGBA-http/` project to the emulator once it starts, else the emulator cannot receive any messages.

## Components

### Base Components
- **MGBAController**: A high-level controller class for the mGBA-http API
- **emerald_party_reader.py**: A script for reading Pokémon party data from Pokémon Emerald
- **test_button_controls.py**: A test script for button control functionality
- **example_movement.py**: An example script that demonstrates character movement

### Vision Navigation Components
- **VisionController**: Enhanced controller with computer vision capabilities via Gemini
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
```

See [VISION_NAVIGATION.md](VISION_NAVIGATION.md) for detailed information about the vision-based navigation system.

### AI Game Agent

The GameAgent provides an LLM-powered thinking and playing loop for Pokémon games:

```python
from game_agent import GameAgent
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Initialize with preferred LLM provider (openai, anthropic, or gemini)
agent = GameAgent(llm_provider="gemini")

# Run for a specific number of steps
agent.run_loop(steps=10)

# Or run indefinitely
# agent.run_loop()
```

Run the test script to try it out:

```bash
# Using the Gemini provider (key from .env file)
./test_game_agent.py

# Or specify a different provider
./test_game_agent.py --provider openai --steps 5
```

See [GAME_AGENT.md](GAME_AGENT.md) for detailed information about the AI-powered game agent system.

## Project Structure

- **mgba_controller.py**: Main controller class for interfacing with mGBA-http
- **vision_controller.py**: Enhanced controller with computer vision capabilities
- **game_agent.py**: AI-powered agent that plays the game through LLM thinking
- **mGBA-http/**: Submodule containing the C# HTTP server for mGBA
- **setup.sh**: Script for setting up the project
- **run.sh**: Script for running the mGBA emulator and HTTP server
- **emerald_party_reader.py**: Example script for reading Pokémon Emerald party data
- **test_button_controls.py**: Test script for button control functionality
- **example_movement.py**: Example script for character movement
- **solve_ice_puzzle.py**: Script for solving ice puzzles with computer vision
- **test_game_agent.py**: Test script for the AI game agent
- **VISION_NAVIGATION.md**: Documentation for the vision-based navigation system
- **GAME_AGENT.md**: Documentation for the AI-powered game agent system

## Pokémon Game Support

Currently tested with:
- Pokémon Emerald

## License

MIT License 