# Vision-Based Navigation for Pokémon Games

This system enables intelligent navigation in Pokémon games by combining memory reading with computer vision powered by Google's Gemini Vision API. This approach overcomes the limitations of pure memory-based navigation, especially for complex puzzles like ice sliding areas.

## How It Works

### 1. Screen Capture
The system captures screenshots of the mGBA emulator window, which are sent to Gemini's Vision API for analysis.

### 2. Scene Analysis
Gemini 1.5 Flash model analyzes the screenshot to identify:
- The current location (town, route, building)
- The player's position
- Obstacles and special tiles (walls, ice, water)
- NPCs and interactable objects
- Potential exits and paths

### 3. Intelligent Decision Making
Based on the visual analysis, the system:
- Determines the best directional input to reach a target
- Handles special mechanics like ice sliding (where players slide until hitting an obstacle)
- Adapts to the current game state without needing pre-programmed routes

### 4. Game Control
The system executes the determined actions using the MGBAController class, pressing the appropriate buttons to navigate through the game.

## Key Components

### `VisionController` Class
The main class combining vision and game control capabilities:
- `capture_screen()`: Takes screenshots of the game window
- `analyze_screen()`: Sends screenshots to Gemini for analysis
- `navigate_complex_area()`: Navigates to a target location
- `solve_ice_puzzle()`: Specially handles ice sliding puzzles
- `identify_game_objects()`: Detects NPCs, items, and other objects
- `follow_route()`: Follows a described route using visual guidance

### Special Puzzle Handling

Ice puzzles in Pokémon games have unique mechanics where the player slides until hitting an obstacle. The system handles this by:

1. Analyzing the entire puzzle layout visually
2. Determining which direction to press based on the sliding mechanics
3. Calculating which obstacles the player will hit when sliding in each direction
4. Choosing the optimal move to make progress toward the exit

## Requirements

- Google Gemini API key (Vision model access)
- Python 3.6+
- mGBA emulator
- PIL (Python Imaging Library)
- google-generativeai package

## Usage Example

```python
from vision_controller import VisionController
from mgba_controller import MGBAController

# Initialize controllers
mgba_controller = MGBAController()
vision_controller = VisionController(
    api_key="YOUR_GEMINI_API_KEY",
    mgba_controller=mgba_controller
)

# Solve an ice puzzle
vision_controller.solve_ice_puzzle(exit_description="ladder")

# Or navigate to a specific destination
vision_controller.navigate_complex_area(
    target_description="Pokemon Center",
    max_steps=20
)
```

## Scripts

- `solve_ice_puzzle.py`: Example script demonstrating how to use the vision controller to solve ice puzzles
- `vision_controller.py`: The main controller class implementing vision-based navigation

## Limitations

- Requires a valid Gemini API key with access to vision models
- The quality of navigation depends on Gemini's ability to correctly analyze game screenshots
- Screenshot capture methods may vary depending on OS and window management systems
- Reaction time is slower than pure memory-based approaches due to API calls 