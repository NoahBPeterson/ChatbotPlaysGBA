#!/usr/bin/env python3
"""
Vision Controller Module

Uses a hybrid approach that prioritizes memory reading for tile maps,
with Gemini's Vision API as a fallback for complex navigation scenarios.
"""

import os
import time
import base64
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import subprocess
import requests
from PIL import Image, ImageGrab
import google.generativeai as genai

# Import providers for consistency with game_agent.py
try:
    import openai
except ImportError:
    openai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from mgba_controller import MGBAController, Button, MemoryDomain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vision_controller')

# Common tile types in Pokémon games
class TileType:
    FLOOR = 0
    WALL = 1
    WATER = 2
    ICE = 3
    LEDGE = 4
    DOOR = 5
    NPC = 6
    GRASS = 7
    TREE = 8
    UNKNOWN = 99

# Memory locations for different Pokémon games (offsets relative to WRAM)
# These would need to be researched/verified for each game
MEMORY_MAPS = {
    "POKEMON EMER": {
        "map_id": 0x3E05C,
        "player_x": 0x3E06E,
        "player_y": 0x3E070,
        "map_width": None,  # Varies by map
        "map_height": None, # Varies by map
        "tile_data_ptr": None, # Varies by map
        "tileset_ptr": None, # Varies by map
    }
}

class VisionController:
    """
    Hybrid controller that prioritizes memory reading for tile maps,
    but falls back to vision capabilities when needed.
    """
    def __init__(self, 
                 api_key: str,
                 provider_name: str = "gemini",
                 mgba_controller: Optional[MGBAController] = None, 
                 mgba_window_title: str = "mGBA"):
        """
        Initialize the vision controller.
        
        Args:
            api_key: API key for the vision model
            provider_name: Name of the provider to use (gemini, openai, anthropic)
            mgba_controller: Existing MGBAController instance or None to create a new one
            mgba_window_title: Window title of the mGBA emulator for screen capture
        """
        # Store API key and provider name
        self.api_key = api_key
        self.provider_name = provider_name.lower()
        
        # Initialize the vision model based on provider
        self._initialize_vision_model()
        
        # Initialize the mGBA controller
        self.controller = mgba_controller if mgba_controller else MGBAController()
        
        # Store mGBA window title for screen capture
        self.mgba_window_title = mgba_window_title
        
        # Navigation memory
        self.known_obstacles = []
        self.current_map = None
        self.last_screenshot_path = None
        self.tile_map = None
        self.player_position = (0, 0)
        
        # Initialize game-specific memory mappings
        self.memory_map = MEMORY_MAPS.get(self.controller.game_title, {})
        if self.memory_map:
            logger.info(f"Loaded memory map for {self.controller.game_title}")
        else:
            logger.warning(f"No memory map available for {self.controller.game_title}")
        
        logger.info("Vision Controller initialized successfully")
    
    def _initialize_vision_model(self):
        """Initialize the vision model based on the provider name."""
        if self.provider_name == "gemini":
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.model_type = "gemini"
            logger.info("Initialized Gemini vision model")
        elif self.provider_name == "openai":
            if openai is None:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
            self.model = openai.OpenAI(api_key=self.api_key)
            self.model_type = "openai"
            logger.info("Initialized OpenAI vision model")
        elif self.provider_name == "anthropic":
            if Anthropic is None:
                raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
            self.model = Anthropic(api_key=self.api_key)
            self.model_type = "anthropic"
            logger.info("Initialized Anthropic vision model")
        elif self.provider_name == "deepseek":
            # DeepSeek doesn't have vision capabilities yet, so we'll fall back to Gemini
            # but get a separate Gemini API key for vision
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                logger.warning("DeepSeek provider doesn't support vision and no GEMINI_API_KEY found. Falling back to basic vision capabilities.")
                self.model_type = "basic"
            else:
                genai.configure(api_key=gemini_api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                self.model_type = "gemini"
                logger.info("Using Gemini vision model with DeepSeek provider")
        else:
            # Default to Gemini if unknown provider
            logger.warning(f"Unknown provider {self.provider_name}, falling back to Gemini")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.model_type = "gemini"
            
    def _generate_vision_content(self, prompt: str, image_path: str) -> str:
        """
        Generate content from vision model based on the provider type.
        
        Args:
            prompt: Text prompt for the vision model
            image_path: Path to the image file
            
        Returns:
            The generated text response
        """
        img = Image.open(image_path)
        
        if self.model_type == "gemini":
            response = self.model.generate_content([prompt, img])
            return response.text
        
        elif self.model_type == "openai":
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            response = self.model.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes game screenshots."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        elif self.model_type == "anthropic":
            # Convert image to JPEG and encode to base64
            # Create a temporary JPEG file
            jpeg_path = image_path
            if not image_path.lower().endswith('.jpg') and not image_path.lower().endswith('.jpeg'):
                temp_jpeg = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                jpeg_path = temp_jpeg.name
                temp_jpeg.close()
                
                # Convert to JPEG
                img = img.convert('RGB')
                img.save(jpeg_path, 'JPEG')
                logger.info(f"Converted image to JPEG at {jpeg_path}")
            
            # Read and encode the JPEG image
            with open(jpeg_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Delete temporary file if created
            if jpeg_path != image_path:
                try:
                    os.unlink(jpeg_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary JPEG file: {e}")
                
            response = self.model.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                    ]}
                ]
            )
            return response.content[0].text
            
        elif self.model_type == "basic":
            # If no vision model is available, return a basic description
            logger.warning("Using basic vision capabilities (no actual image processing)")
            
            # Extract filename and dimensions as basic info
            filename = os.path.basename(image_path)
            width, height = img.size
            
            # Return a generic description
            return f"This appears to be a screenshot from a Pokémon game ({width}x{height} pixels). " \
                   f"Without advanced vision capabilities, I can't analyze the specific details. " \
                   f"You may want to press common navigation buttons like A, START, or directional buttons to progress."
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def read_map_from_memory(self) -> Dict[str, Any]:
        """
        Attempt to read the current map data from memory.
        
        Returns:
            Dict containing map data or empty dict if unsuccessful
        """
        if not self.memory_map:
            logger.warning("No memory mappings available for this game")
            return {}
        
        try:
            # Read basic map information
            map_data = {}
            
            # Read current map ID
            if "map_id" in self.memory_map:
                map_id = self.controller.read_word(self.memory_map["map_id"], "wram")
                map_data["map_id"] = map_id
                logger.info(f"Current map ID: {map_id}")
            
            # Read player position
            if "player_x" in self.memory_map and "player_y" in self.memory_map:
                player_x = self.controller.read_word(self.memory_map["player_x"], "wram")
                player_y = self.controller.read_word(self.memory_map["player_y"], "wram")
                self.player_position = (player_x, player_y)
                map_data["player_position"] = self.player_position
                logger.info(f"Player position: ({player_x}, {player_y})")
            
            # To read the actual tile map, we would need more complex logic
            # that depends on the specific game's memory layout
            # This is a simplified placeholder that would need game-specific implementation
            
            return map_data
        
        except Exception as e:
            logger.error(f"Error reading map from memory: {e}")
            return {}
    
    def build_tile_map(self, vision_fallback: bool = True) -> List[List[int]]:
        """
        Build a 2D grid representing the current map's tiles.
        
        Args:
            vision_fallback: Whether to fall back to vision if memory reading fails
            
        Returns:
            2D list of tile types (using TileType constants)
        """
        # Try to read map data from memory first
        map_data = self.read_map_from_memory()
        
        # If we have enough data from memory, build the tile map
        if map_data and "map_width" in map_data and "map_height" in map_data and "tile_data" in map_data:
            logger.info("Building tile map from memory data")
            width = map_data["map_width"]
            height = map_data["map_height"]
            tile_data = map_data["tile_data"]
            
            # Convert 1D tile data to 2D grid
            tile_map = []
            for y in range(height):
                row = []
                for x in range(width):
                    index = y * width + x
                    tile_type = tile_data[index] if index < len(tile_data) else TileType.UNKNOWN
                    row.append(tile_type)
                tile_map.append(row)
            
            self.tile_map = tile_map
            return tile_map
        
        # If memory reading failed or isn't implemented, fall back to vision
        if vision_fallback:
            logger.info("Falling back to vision for tile map")
            return self.build_tile_map_from_vision()
        else:
            logger.warning("Failed to build tile map and vision fallback disabled")
            return [[TileType.UNKNOWN]]
    
    def build_tile_map_from_vision(self) -> List[List[int]]:
        """
        Use vision to build a tile map when memory reading fails.
        
        Returns:
            List[List[int]]: 2D grid of tile types
        """
        screenshot_path = self.capture_screen()
        
        prompt = """
        Analyze this Pokémon game screenshot and create a grid representing the walkable area.
        
        For each tile, indicate:
        - 0 for floor/walkable space
        - 1 for walls/solid obstacles
        - 2 for water
        - 3 for ice
        - 4 for ledges
        - 5 for doors
        - 6 for NPCs
        - 7 for grass
        - 8 for trees
        - 99 for unknown/unclear
        
        Return only a 10x10 grid centered on the player if possible. Format your response 
        as a grid of numbers, exactly like this:
        [[0,0,1,1,1,1,0,0,0,0],
         [0,0,0,1,1,0,0,0,0,0],
         ...and so on for 10 rows]
        """
        
        try:
            response_text = self._generate_vision_content(prompt, screenshot_path)
            
            # Extract the grid from the response
            grid_text = response_text
            
            # Look for grid between triple backticks if present
            if "```" in grid_text:
                grid_text = grid_text.split("```")[1].split("```")[0].strip()
                # Remove any language identifier like "json"
                if grid_text.startswith("json") or grid_text.startswith("python"):
                    grid_text = grid_text.split("\n", 1)[1]
            
            # Try to clean up and parse the text
            try:
                # First attempt: Clean up and parse as JSON
                grid_text = grid_text.replace(" ", "").replace("\n", "")
                grid = json.loads(grid_text)
                
                # Validate the grid
                if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
                    logger.warning(f"Invalid grid format returned from vision: not a 2D list")
                    raise ValueError("Not a valid 2D grid")
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Couldn't parse grid as JSON, falling back to manual parsing: {e}")
                
                # Second attempt: Manual parsing
                # Try to extract rows with regex
                import re
                rows = re.findall(r'\[\s*(\d+(?:\s*,\s*\d+)+)\s*\]', grid_text)
                
                if not rows:
                    logger.warning("Couldn't find valid rows in the grid text")
                    # Create a default 3x3 grid with player in center
                    return [[99, 99, 99], [99, 0, 99], [99, 99, 99]]
                
                # Convert text rows to number lists
                grid = []
                for row in rows:
                    # Split by comma and convert to integers
                    try:
                        numeric_row = [int(num.strip()) for num in row.split(',')]
                        grid.append(numeric_row)
                    except ValueError:
                        logger.warning(f"Invalid row format: {row}")
                        continue
                
                # Check if we have a valid grid now
                if not grid or not all(len(row) > 0 for row in grid):
                    logger.warning("Failed to parse a valid grid with manual parsing")
                    return [[99, 99, 99], [99, 0, 99], [99, 99, 99]]
            
            logger.info(f"Successfully built tile map from vision: {len(grid)}x{len(grid[0]) if grid else 0}")
            return grid
            
        except Exception as e:
            logger.error(f"Failed to build tile map from vision: {e}")
            # Return a minimal 3x3 grid with player in center as fallback
            return [[99, 99, 99], [99, 0, 99], [99, 99, 99]]
    
    def get_tile_at(self, x: int, y: int) -> int:
        """
        Get the tile type at the specified coordinates.
        
        Args:
            x: X coordinate relative to the map origin
            y: Y coordinate relative to the map origin
            
        Returns:
            TileType constant
        """
        # Ensure we have a tile map
        if self.tile_map is None:
            self.build_tile_map()
        
        # Calculate coordinates relative to the tile map
        if self.player_position and self.tile_map:
            player_x, player_y = self.player_position
            # Convert absolute coordinates to grid coordinates
            grid_x = len(self.tile_map[0]) // 2 + (x - player_x)
            grid_y = len(self.tile_map) // 2 + (y - player_y)
            
            # Check bounds
            if 0 <= grid_y < len(self.tile_map) and 0 <= grid_x < len(self.tile_map[grid_y]):
                return self.tile_map[grid_y][grid_x]
        
        return TileType.UNKNOWN

    def capture_screen(self, save_path: Optional[str] = None) -> str:
        """
        Capture a screenshot of the mGBA window using the mGBA-http API.
        
        Args:
            save_path: Optional path to save the screenshot
            
        Returns:
            str: Path to the saved screenshot
        """
        if save_path is None:
            # Create a temporary file if no path provided
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f"mgba_screenshot_{int(time.time())}.png")
        
        try:
            # Use the mGBA-http API to capture a screenshot
            # This is much more reliable than OS-level screen capture
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Make the API call to capture a screenshot
            logger.info(f"Capturing screenshot via mGBA-http API to {save_path}")
            response = requests.post(
                f"{self.controller.base_url}/core/screenshot",
                params={"path": os.path.abspath(save_path)},
                timeout=self.controller.timeout
            )
            response.raise_for_status()
            
            # Check if the file was created
            if os.path.exists(save_path):
                logger.info(f"Screenshot saved to {save_path}")
                self.last_screenshot_path = save_path
                return save_path
            else:
                logger.error(f"Screenshot file {save_path} was not created")
                raise FileNotFoundError(f"Screenshot file {save_path} was not created")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error capturing screenshot via API: {e}")
            
            # Fall back to OS-level screen capture if API fails
            logger.warning("Falling back to OS-level screen capture")
            
            # On macOS, use screencapture command to capture the mGBA window
            if os.name == 'posix':
                try:
                    # Try using screencapture on macOS
                    subprocess.run([
                        'screencapture', 
                        '-l', f"$(osascript -e 'tell app \"System Events\" to id of window \"{self.mgba_window_title}\" of process \"mGBA\"')",
                        save_path
                    ], check=True)
                    logger.info(f"Screenshot saved to {save_path} using screencapture")
                    self.last_screenshot_path = save_path
                    return save_path
                except subprocess.SubprocessError:
                    logger.warning("Failed to capture window with screencapture")
            
            # Fallback: capture the entire screen
            screen = ImageGrab.grab()
            screen.save(save_path)
            logger.info(f"Captured full screen to {save_path}")
            self.last_screenshot_path = save_path
            return save_path
    
    def analyze_screen(self, screenshot_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a screenshot using the vision model.
        
        Args:
            screenshot_path: Path to the screenshot or None to capture a new one
            
        Returns:
            Dict: Analysis results
        """
        # First try to get information from memory
        memory_data = self.read_map_from_memory()
        
        # If we got meaningful data from memory, prioritize that
        if memory_data and "map_id" in memory_data:
            logger.info("Using memory data for screen analysis")
            
            # Try to build a more complete map from memory
            self.tile_map = self.build_tile_map(vision_fallback=True)
            
            # If we don't have a memory map or it failed, fall back to vision
            if not self.tile_map:
                logger.warning("Failed to build tile map, falling back to vision")
        
        # For the detailed analysis, we'll use vision regardless
        # First capture a screenshot if one wasn't provided
        if not screenshot_path:
            screenshot_path = self.capture_screen()
        
        prompt = """
        You are analyzing a screenshot from a Pokémon game. Describe in detail:
        1. Where the player is (town, route, building, etc.)
        2. What NPCs are visible
        3. Any important objects or features
        4. Available paths/exits
        5. Any text visible on screen
        
        Keep your response clear and concise, focusing on factual observations.
        """
        
        try:
            response_text = self._generate_vision_content(prompt, screenshot_path)
            logger.info(f"Successfully analyzed screen: {response_text[:50]}...")
            
            # Parse the response into a more structured format
            analysis = {
                "description": response_text,
                "location_type": self._determine_location_type(response_text),
                "objects": [],
                "npcs": [],
                "exits": [],
                "player_position": self.player_position,
                "map_id": memory_data.get("map_id", 0) if memory_data else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing screen: {e}")
            return {
                "description": "Analysis failed due to an error",
                "location_type": "unknown",
                "objects": [],
                "npcs": [],
                "exits": [],
                "player_position": self.player_position,
                "map_id": memory_data.get("map_id", 0) if memory_data else 0
            }
    
    def _determine_location_type(self, description: str) -> str:
        """Determine the location type from the description."""
        description_lower = description.lower()
        
        # Special case for title screen
        if "title screen" in description_lower:
            return "title_screen"
        elif "town" in description_lower or "city" in description_lower:
            return "town"
        elif "route" in description_lower:
            return "route"
        elif "building" in description_lower or "center" in description_lower or "mart" in description_lower:
            return "building"
        elif "cave" in description_lower:
            return "cave"
        else:
            return "unknown"
    
    def navigate_complex_area(self, target_description: str, max_steps: int = 10) -> bool:
        """
        Navigate through a complex area using vision guidance.
        
        Args:
            target_description: Description of the target (e.g., "exit", "Pokemon Center")
            max_steps: Maximum number of steps to take
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Navigating to {target_description}")
        
        for step in range(max_steps):
            logger.info(f"Navigation step {step+1}/{max_steps}")
            
            # Capture the current state
            screenshot_path = self.capture_screen()
            
            # First, analyze what we're seeing
            analysis = self.analyze_screen(screenshot_path)
            
            # Then ask for the next move
            prompt = f"""
            I'm trying to navigate to: {target_description}
            
            Based on this screenshot, tell me the SINGLE BEST button to press next.
            Respond with ONLY ONE of: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT
            
            Choose the direction that will most likely lead me to my destination.
            """
            
            try:
                response_text = self._generate_vision_content(prompt, screenshot_path)
                
                # Extract just the button name
                button_text = response_text.strip().upper()
                if "UP" in button_text:
                    button = "UP"
                elif "DOWN" in button_text:
                    button = "DOWN"
                elif "LEFT" in button_text:
                    button = "LEFT"
                elif "RIGHT" in button_text:
                    button = "RIGHT"
                elif "A" in button_text:
                    button = "A"
                elif "B" in button_text:
                    button = "B"
                elif "START" in button_text:
                    button = "START"
                elif "SELECT" in button_text:
                    button = "SELECT"
                else:
                    button = "A"  # Default to A if no clear direction
                
                logger.info(f"Navigation move: {button}")
                
                # Execute the move
                self.execute_text_move(button)
                
                # Check if we've reached the destination
                check_prompt = f"""
                Have I reached {target_description} yet?
                Answer with only YES or NO.
                """
                
                check_response = self._generate_vision_content(check_prompt, self.capture_screen())
                
                if "YES" in check_response.upper():
                    logger.info(f"Successfully navigated to {target_description}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error during navigation: {e}")
                
        logger.warning(f"Failed to navigate to {target_description} in {max_steps} steps")
        return False
    
    def execute_text_move(self, move_text: str) -> None:
        """
        Execute a move described in text.
        
        Args:
            move_text: Text describing the move (e.g., "go up", "press A")
        """
        if not move_text or not isinstance(move_text, str):
            logger.warning(f"Invalid move text: {move_text}")
            return
        
        # Map simple text directions to buttons
        direction_keywords = {
            "up": Button.UP,
            "north": Button.UP,
            "down": Button.DOWN,
            "south": Button.DOWN,
            "left": Button.LEFT,
            "west": Button.LEFT,
            "right": Button.RIGHT,
            "east": Button.RIGHT,
            "press a": Button.A,
            "button a": Button.A,
            "press b": Button.B,
            "button b": Button.B,
            "start": Button.START,
            "select": Button.SELECT
        }
        
        # Convert the move text to lowercase for easier matching
        move_text = move_text.lower()
        
        # First check if the move is just a single button letter (like "a" or "b")
        if move_text == "a":
            self.controller.press_button(Button.A)
            return
        elif move_text == "b":
            self.controller.press_button(Button.B)
            return
        
        # Then look for longer texts containing button names
        for keyword, button in direction_keywords.items():
            if keyword in move_text:
                self.controller.press_button(button)
                return
        
        logger.warning(f"Couldn't parse move text: {move_text}")
  
    def identify_game_objects(self, object_type: str = "all") -> List[Dict[str, Any]]:
        """
        Identify specific types of objects in the current screen.
        
        Args:
            object_type: Type of objects to identify ("npcs", "items", "pokemon", "all")
            
        Returns:
            List[Dict]: Information about identified objects
        """
        # Try to get object info from memory first
        # This would require game-specific implementation
        memory_objects = []
        
        # Fall back to vision if memory approach doesn't yield results
        if not memory_objects:
            self.capture_screen()
            
            # Create a specific prompt for object detection
            prompt = f"""
            Identify all {object_type} visible in this Pokémon game screenshot.
            For each {object_type}, provide:
            1. Type or name (if recognizable)
            2. Position (approximate x,y coordinates or description)
            3. Any distinguishing features
            
            Format your response as a JSON array of objects.
            """
            
            img = Image.open(self.last_screenshot_path)
            response = self.model.generate_content([prompt, img])
            
            # Extract JSON from response
            result = response.text
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            try:
                objects = json.loads(result)
                logger.info(f"Identified {len(objects)} {object_type} in screen")
                return objects
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON response from vision API")
                return []
        
        return memory_objects
    
    def follow_route(self, route_description: str, max_steps: int = 20) -> bool:
        """
        Follow a described route using vision guidance.
        
        Args:
            route_description: Description of the route to follow
            max_steps: Maximum number of steps to take
            
        Returns:
            bool: True if successfully followed the route, False otherwise
        """
        logger.info(f"Following route: {route_description}")
        
        for step in range(max_steps):
            logger.info(f"Route following step {step+1}/{max_steps}")
            
            # Capture current state
            screenshot_path = self.capture_screen()
            
            # Ask for the next action
            prompt = f"""
            I need to follow this route: {route_description}
            
            Based on the current screenshot, what is the SINGLE NEXT ACTION I should take?
            Respond with one of:
            - MOVE UP
            - MOVE DOWN
            - MOVE LEFT
            - MOVE RIGHT
            - PRESS A
            - PRESS B
            - WAIT
            - DESTINATION REACHED
            
            Keep your response to ONLY those exact phrases without explanation.
            """
            
            try:
                response_text = self._generate_vision_content(prompt, screenshot_path)
                action = response_text.strip().upper()
                
                logger.info(f"Route action: {action}")
                
                if "DESTINATION" in action or "REACHED" in action:
                    logger.info(f"Destination reached: {route_description}")
                    return True
                    
                # Execute the action
                if "MOVE UP" in action:
                    self.controller.press_button(Button.UP)
                elif "MOVE DOWN" in action:
                    self.controller.press_button(Button.DOWN)
                elif "MOVE LEFT" in action:
                    self.controller.press_button(Button.LEFT)
                elif "MOVE RIGHT" in action:
                    self.controller.press_button(Button.RIGHT)
                elif "PRESS A" in action:
                    self.controller.press_button(Button.A)
                elif "PRESS B" in action:
                    self.controller.press_button(Button.B)
                elif "WAIT" in action:
                    pass
                
                # Wait for the action to complete
                time.sleep(1)
                
                # Periodically check if we've reached the destination
                if step % 5 == 0 or "DESTINATION" in action or "REACHED" in action:
                    self.capture_screen()
                    status_prompt = f"Have I reached the destination of {route_description}? Answer YES or NO only."
                    
                    status_response = self._generate_vision_content(status_prompt, self.last_screenshot_path)
                    
                    if "YES" in status_response.upper():
                        logger.info(f"Destination reached: {route_description}")
                        return True
            
            except Exception as e:
                logger.error(f"Error during route following: {e}")
            
        logger.warning(f"Failed to complete route within {max_steps} steps")
        return False


# Example usage
if __name__ == "__main__":
    # You would need to provide your Gemini API key
    API_KEY = os.environ.get("GEMINI_API_KEY") 
    
    if not API_KEY:
        print("Please set GEMINI_API_KEY environment variable")
        exit(1)
    
    vision_controller = VisionController(API_KEY)
    
    # Capture a screenshot
    screenshot_path = vision_controller.capture_screen()
    print(f"Screenshot saved to: {screenshot_path}")
    
    # Try to read map data from memory first
    map_data = vision_controller.read_map_from_memory()
    if map_data:
        print(f"Successfully read map data from memory: {map_data}")
    else:
        print("Couldn't read map data from memory, falling back to vision")
    
    # Build a tile map (using vision as fallback if needed)
    tile_map = vision_controller.build_tile_map()
    print(f"Built a tile map with {len(tile_map)} rows")
    
    # Analyze the current screen with the hybrid approach
    analysis = vision_controller.analyze_screen()
    print(f"Current location: {analysis['location']}")
    print(f"Player position: {analysis['player_position']}")
    
    if 'obstacles' in analysis and analysis['obstacles']:
        print(f"Obstacles: {', '.join(analysis['obstacles'])}")
    
    if 'recommended_moves' in analysis and analysis['recommended_moves']:
        print(f"Recommended moves: {', '.join(analysis['recommended_moves'])}") 