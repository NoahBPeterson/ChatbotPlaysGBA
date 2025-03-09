# mGBA-http Python Utilities

This repository contains Python utilities for interfacing with mGBA emulator via its HTTP API.

## Overview

These tools allow you to control the mGBA emulator and access game memory through a simple API. They're particularly useful for tasks like:

- Programmatically controlling game inputs
- Reading and writing to memory
- Building automation tools and bots
- Developing AI agents that can play games

## Setup

### Prerequisites

- Python 3.6+
- [.NET SDK](https://dotnet.microsoft.com/download) (for building mGBA-http)
- [mGBA](https://mgba.io/downloads.html) installed and in your PATH
- git

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

## Components

- **MGBAController**: A high-level controller class for the mGBA-http API
- **emerald_party_reader.py**: A script for reading Pokémon party data from Pokémon Emerald
- **test_button_controls.py**: A test script for button control functionality
- **example_movement.py**: An example script that demonstrates character movement

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

## Project Structure

- **mgba_controller.py**: Main controller class for interfacing with mGBA-http
- **mGBA-http/**: Submodule containing the C# HTTP server for mGBA
- **setup.sh**: Script for setting up the project
- **run.sh**: Script for running the mGBA emulator and HTTP server
- **emerald_party_reader.py**: Example script for reading Pokémon Emerald party data
- **test_button_controls.py**: Test script for button control functionality
- **example_movement.py**: Example script for character movement

## Pokémon Game Support

Currently tested with:
- Pokémon Emerald

## License

MIT License 