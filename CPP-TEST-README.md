# Claude Plays Pokémon (CPP) - Test Harness

This test harness provides a way to verify the mGBA-http integration, ensuring all components are working correctly.

## Prerequisites

1. mGBA emulator installed and running
2. mGBA-http server built and running
3. mGBASocketServer.lua script loaded in mGBA
4. A ROM loaded in mGBA
5. Python 3.12 or higher

## Setup

### Setting up the Python environment

We recommend using `uv` for Python environment management:

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

Alternatively, if you don't have `uv` installed:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Test Harness

The test harness is a Python script that sends various API requests to the mGBA-http server and verifies the responses.

```bash
# Run all tests
./test_mgba_http.py

# Run a specific test
./test_mgba_http.py --test connection
./test_mgba_http.py --test button
./test_mgba_http.py --test memory
./test_mgba_http.py --test sequence
./test_mgba_http.py --test combined
./test_mgba_http.py --test swagger

# Use a different API base URL (if you changed the port)
./test_mgba_http.py --url http://localhost:6969
```

## Available Tests

1. **connection** - Tests basic connectivity by requesting the ROM title
2. **button** - Tests pressing a single button (A)
3. **memory** - Tests reading from the game's memory
4. **sequence** - Tests a sequence of button presses (Up followed by A)
5. **combined** - Tests pressing multiple buttons simultaneously (Up+A)
6. **swagger** - Tests access to the Swagger API documentation

## Troubleshooting

If the tests fail, ensure:

1. mGBA emulator is running
2. mGBA-http server is running
3. mGBASocketServer.lua script is loaded in mGBA
4. A ROM is loaded in mGBA
5. The correct port is being used (default: 5000)
6. Your virtual environment is activated and all dependencies are installed

## Integration with LLM-Based System

This test harness serves as a technical foundation for the "Claude Plays Pokémon" project. Once these tests pass, you can be confident that an LLM-based system can effectively interact with the emulator through the HTTP API.

The next steps in development would be:
1. Create a Python module that wraps these API calls into higher-level functions
2. Develop the vision system for interpreting game state from screenshots
3. Build the decision-making system that will determine what buttons to press
4. Implement the knowledge management system for game state tracking 