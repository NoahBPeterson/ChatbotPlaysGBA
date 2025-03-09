# Claude Plays Pokémon (CPP) - Architecture Plan

## 1. Core Components

1. **Emulator Integration**
   - Game ROM interface
   - Screenshot capture system
   - Button input mechanism
   - Save state management
   - RAM state reading capabilities

2. **Knowledge Management System**
   - Structured database
   - Section-based organization (game mechanics, locations, NPCs, Pokémon)
   - Update/retrieval mechanisms

3. **Vision System**
   - OCR for text recognition
   - Scene classification
   - Object detection (NPCs, items, obstacles)
   - Map awareness

4. **Decision Making Engine**
   - Strategic planning module
   - Tactical battle system
   - Exploration algorithm
   - Progression tracking

## 2. Tool Suite

1. **`update_knowledge_base`**
   - Parameters: section_id, content, operation (add/delete/modify)
   - Returns: Confirmation of changes

2. **`use_emulator`**
   - Parameters: button_sequence, duration, wait_for_frame_change
   - Returns: Screenshot, RAM state, overlay analysis

3. **`navigator`**
   - Parameters: destination, avoid_trainers (boolean)
   - Returns: Success/failure message, path taken

4. **`battle_manager`**
   - Parameters: battle_strategy, priority_targets
   - Returns: Battle outcome, experience gained, moves learned

5. **`twitch_reader`**
   - Parameters: scan_duration, filter_criteria
   - Returns: Filtered suggestions from chat

## 3. Learning Systems

1. **Self-Improvement Loop**
   - Performance metrics tracker
   - Strategy revision system
   - Failure analysis module

2. **Curriculum Learning**
   - Progression stages (route navigation → trainer battles → gym battles → Elite Four)
   - Difficulty scaling
   - Achievement unlocking

3. **World Model**
   - Game state representation
   - Action-consequence predictor
   - Environment simulator for planning

## 4. Memory Management

1. **Short-term Memory**
   - Current objective
   - Recent interactions
   - Immediate environment

2. **Task-oriented Memory**
   - Quest tracking
   - Item inventory
   - Team composition and status

3. **Long-term Strategy**
   - Game map understanding
   - Type effectiveness knowledge
   - Key progression requirements

## 5. System Architecture

1. **Observation Layer**
   - Game state parsing
   - Visual information extraction
   - Text interpretation

2. **Planning Layer**
   - High-level goal setting
   - Path planning
   - Battle preparation

3. **Execution Layer**
   - Button sequence generation
   - Timing management
   - Error handling

4. **Reflection Layer**
   - Success/failure evaluation
   - Strategy adjustment
   - Knowledge base updates

## 6. Implementation Progress

### mGBA-http Integration

We have successfully cloned and built the mGBA-http project for macOS ARM64 architecture. This will allow us to interface with the mGBA emulator programmatically through HTTP requests, which is essential for our Claude Plays Pokémon implementation.

**Build Details:**
- Successfully cloned [nikouu/mGBA-http](https://github.com/nikouu/mGBA-http.git)
- Built the project for Apple Silicon (ARM64) architecture
- Generated a self-contained executable that doesn't require .NET runtime installation
- Included the necessary mGBASocketServer.lua script

### Test Harness Development

Created a Python-based test harness (`test_mgba_http.py`) to verify the functionality of the mGBA-http API. The test harness includes:

**Test Coverage:**
- Basic connection testing (retrieving ROM title)
- Button press functionality (single button, sequences, and combinations)
- Memory reading
- Swagger API documentation access

**Features:**
- Colorized console output with clear pass/fail indicators
- Detailed response information for debugging
- Command-line arguments for selecting specific tests
- API base URL configuration
- Comprehensive error handling and reporting
- Configurable timeouts to prevent hanging on unresponsive endpoints
- Support for all button press types (single, sequence, simultaneous)

**Improvements:**
- Added proper timeout handling to prevent the test harness from hanging indefinitely
- Updated API endpoints to match the actual mGBA-http implementation
- Corrected parameter passing for button press endpoints (using query parameters instead of JSON body)
- Added support for simultaneous button presses via the tapmany endpoint

### High-Level Controller Module

Developed a high-level Python controller module (`mgba_controller.py`) that provides a clean, object-oriented interface to the mGBA-http API:

**Features:**
- Button press abstraction with support for single, sequential, and simultaneous presses
- Memory reading/writing with support for different data sizes (byte, word, double word)
- Game state management (save/load states, reset)
- Logging capabilities
- Higher-level game actions (e.g., "open_menu", "select_option")
- Proper error handling and timeouts
- Type hints and comprehensive documentation

**Next Steps:**
1. Create a vision system module for interpreting game state from screenshots
2. Build the decision-making system that will determine what buttons to press
3. Implement the knowledge management system for game state tracking
4. Integrate with a screenshot capture system to enable vision-based control 