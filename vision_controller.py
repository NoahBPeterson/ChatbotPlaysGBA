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
                 mgba_controller: Optional[MGBAController] = None, 
                 mgba_window_title: str = "mGBA"):
        """
        Initialize the vision controller.
        
        Args:
            api_key: Gemini API key
            mgba_controller: Existing MGBAController instance or None to create a new one
            mgba_window_title: Window title of the mGBA emulator for screen capture
        """
        # Set up Gemini API access
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
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
        Use Gemini Vision to analyze the screen and build a tile map.
        
        Returns:
            2D list of tile types (using TileType constants)
        """
        self.capture_screen()
        img = Image.open(self.last_screenshot_path)
        
        # Create a specialized prompt for tile mapping
        prompt = """
        Analyze this Pokémon game screenshot and identify the tile-based structure of the map.
        
        1. Identify the player's position within the grid
        2. Map out a grid of tiles around the player (approximately 7x7 tiles)
        3. Classify each tile as one of:
           - FLOOR: Normal walkable tiles
           - WALL: Impassable barriers
           - WATER: Water tiles
           - ICE: Ice tiles that cause sliding
           - LEDGE: One-way passages
           - DOOR: Entrances/exits
           - NPC: Characters
           - GRASS: Wild Pokémon encounters
           - TREE: Cuttable trees
        
        Return the data as a JSON with:
        1. player_position: [x, y] (where [0,0] is top-left of your grid)
        2. grid: A 2D array where each cell contains the tile type
        3. objects: Any special objects visible (NPCs, items) with their coordinates
        
        For example:
        {
          "player_position": [3, 3],
          "grid": [
            ["WALL", "WALL", "WALL", "DOOR", "WALL", "WALL", "WALL"],
            ["WALL", "FLOOR", "FLOOR", "FLOOR", "FLOOR", "FLOOR", "WALL"],
            ...
          ],
          "objects": [
            {"type": "NPC", "position": [5, 2], "description": "woman in red"}
          ]
        }
        """
        
        try:
            response = self.model.generate_content([prompt, img])
            
            # Extract the JSON from the response
            result = response.text
            
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            data = json.loads(result)
            
            # Convert the text-based grid to numeric tile types
            type_mapping = {
                "FLOOR": TileType.FLOOR,
                "WALL": TileType.WALL,
                "WATER": TileType.WATER,
                "ICE": TileType.ICE,
                "LEDGE": TileType.LEDGE,
                "DOOR": TileType.DOOR,
                "NPC": TileType.NPC,
                "GRASS": TileType.GRASS,
                "TREE": TileType.TREE
            }
            
            numeric_grid = []
            for row in data["grid"]:
                numeric_row = []
                for cell in row:
                    numeric_row.append(type_mapping.get(cell, TileType.UNKNOWN))
                numeric_grid.append(numeric_row)
            
            # Store the player position
            if "player_position" in data:
                self.player_position = tuple(data["player_position"])
            
            # Store the complete tile map
            self.tile_map = numeric_grid
            return numeric_grid
            
        except Exception as e:
            logger.error(f"Error building tile map from vision: {e}")
            # Return a minimal 3x3 unknown grid centered on the player
            return [
                [TileType.UNKNOWN, TileType.UNKNOWN, TileType.UNKNOWN],
                [TileType.UNKNOWN, TileType.FLOOR, TileType.UNKNOWN],
                [TileType.UNKNOWN, TileType.UNKNOWN, TileType.UNKNOWN]
            ]
    
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
    
    def predict_ice_slide_endpoint(self, start_x: int, start_y: int, direction: str) -> Tuple[int, int]:
        """
        Predict where the player will end up after sliding on ice in a given direction.
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            direction: "up", "down", "left", or "right"
            
        Returns:
            Tuple of (end_x, end_y)
        """
        # Ensure we have an up-to-date tile map
        if self.tile_map is None:
            self.build_tile_map()
        
        # Define direction vectors
        direction_vectors = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0)
        }
        
        # Get the direction vector
        dx, dy = direction_vectors.get(direction.lower(), (0, 0))
        
        # Start at the initial position
        x, y = start_x, start_y
        
        # Keep sliding until hitting a non-ice tile
        max_steps = 20  # Safety limit
        steps = 0
        while steps < max_steps:
            # Move one tile in the direction
            next_x = x + dx
            next_y = y + dy
            
            # Check the next tile
            next_tile = self.get_tile_at(next_x, next_y)
            
            # If it's a wall or other obstacle, stop at the current position
            if next_tile in [TileType.WALL, TileType.TREE, TileType.NPC]:
                return (x, y)
            
            # If it's not ice, stop at the next position
            if next_tile != TileType.ICE:
                return (next_x, next_y)
            
            # Otherwise, continue sliding
            x, y = next_x, next_y
            steps += 1
        
        # If we've reached the step limit, just return the final position
        return (x, y)
    
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
        Analyze a screenshot using Gemini's Vision API.
        
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
            self.build_tile_map(vision_fallback=False)
            
            # We'll still need to use vision for some things like determining the location name
            # and recommended moves, but we'll integrate the memory data
        
        # Capture a new screenshot if none provided
        if screenshot_path is None:
            if self.last_screenshot_path and os.path.exists(self.last_screenshot_path):
                screenshot_path = self.last_screenshot_path
            else:
                screenshot_path = self.capture_screen()
        
        # Load the image
        img = Image.open(screenshot_path)
        
        # Craft the prompt, mentioning we prefer structured tile-based information
        prompt = """
        Analyze this Pokémon game screenshot and provide the following information:
        1. Current location (town/route/building)
        2. Player position (coordinates or description)
        3. Visible obstacles or special tiles (e.g., ledges, water, ice)
        4. Visible NPCs or objects
        5. Recommended next move(s) if navigating to exit or key location
        
        IMPORTANT: If this is a tile-based puzzle (especially an ice puzzle), please:
        - Provide coordinates in a grid format where each cell is one traversable tile
        - Specify the exact coordinates of walls, obstacles and ice tiles
        - Give player position as (x,y) coordinates in this grid
        
        Format your response as JSON with these fields:
        {
          "location": "...",
          "player_position": "...",
          "obstacles": [...],
          "npcs": [...],
          "special_tiles": [...],
          "exits": [...],
          "tile_grid": [[]], // Include if it's a visible puzzle area
          "recommended_moves": [...]
        }
        """
        
        try:
            response = self.model.generate_content([prompt, img])
            
            # Extract the JSON from the response
            result = response.text
            
            # Clean up the result to make sure it's valid JSON
            # Look for JSON between triple backticks if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(result)
            
            # Integrate memory data if available
            if memory_data:
                if "player_position" in memory_data:
                    analysis["player_position_memory"] = memory_data["player_position"]
                if "map_id" in memory_data:
                    analysis["map_id"] = memory_data["map_id"]
            
            logger.info(f"Successfully analyzed screen: {analysis['location']}")
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing screen: {e}")
            return {
                "error": str(e),
                "location": "unknown",
                "player_position": "unknown",
                "obstacles": [],
                "recommended_moves": []
            }
    
    def navigate_complex_area(self, target_description: str, max_steps: int = 10) -> bool:
        """
        Use a hybrid approach to navigate through a complex area.
        Prioritizes memory reading for tile data, with vision as backup.
        
        Args:
            target_description: Description of the target location
            max_steps: Maximum number of steps to attempt
            
        Returns:
            bool: True if navigation was successful
        """
        logger.info(f"Attempting to navigate to: {target_description}")
        
        for step in range(max_steps):
            # First try to read map data from memory
            map_data = self.read_map_from_memory()
            
            # Build a tile map (will use vision as fallback if memory reading fails)
            tile_map = self.build_tile_map()
            
            # Now get additional context from vision
            analysis = self.analyze_screen()
            
            # Check if we've reached the target
            if target_description.lower() in analysis["location"].lower():
                logger.info(f"Reached target: {target_description}")
                return True
            
            # Determine next move using a hybrid approach
            next_move = self.determine_next_move(analysis, target_description, tile_map)
            
            if next_move:
                logger.info(f"Step {step+1}/{max_steps}: Taking move: {next_move}")
                
                # Execute the move
                self.execute_move(next_move)
                
                # Wait for movement to complete
                time.sleep(1)
            else:
                logger.warning("Couldn't determine next move")
                # Try following recommended moves from vision as fallback
                if "recommended_moves" in analysis and analysis["recommended_moves"]:
                    fallback_move = analysis["recommended_moves"][0]
                    logger.info(f"Using vision fallback move: {fallback_move}")
                    self.execute_text_move(fallback_move)
                    time.sleep(1)
                else:
                    # Last resort: try a random direction
                    self.controller.press_button(Button.UP)
                    time.sleep(1)
            
            # Wait a bit before next analysis
            time.sleep(0.5)
        
        logger.warning(f"Failed to reach {target_description} within {max_steps} steps")
        return False
    
    def determine_next_move(self, analysis: Dict[str, Any], target: str, tile_map: List[List[int]]) -> str:
        """
        Determine the next move based on hybrid analysis.
        
        Args:
            analysis: Screen analysis from vision
            target: Target description
            tile_map: 2D grid of tile types
            
        Returns:
            str: Direction to move ("up", "down", "left", "right", "a", "b")
        """
        # If we're on ice, use specialized ice puzzle logic
        if any(tile for tile in analysis.get("special_tiles", []) if "ice" in str(tile).lower()):
            return self.determine_ice_puzzle_move(analysis, target)
        
        # If we have "tile_grid" from vision, use that for pathfinding
        if "tile_grid" in analysis and analysis["tile_grid"]:
            # Simple greedy pathing toward exits or points of interest
            if "exits" in analysis and analysis["exits"]:
                for exit_info in analysis["exits"]:
                    if target.lower() in str(exit_info).lower():
                        # Found our target exit, try to path to it
                        if "position" in exit_info:
                            exit_pos = exit_info["position"]
                            # Compare with player position and determine direction
                            player_pos = analysis.get("player_position", "center")
                            return self.path_to_position(player_pos, exit_pos)
        
        # If we have recommended moves from vision, use the first one
        if "recommended_moves" in analysis and analysis["recommended_moves"]:
            return analysis["recommended_moves"][0].lower()
        
        # Default to moving toward doors or openings
        for y, row in enumerate(tile_map):
            for x, tile in enumerate(row):
                if tile == TileType.DOOR:
                    # Found a door, path to it
                    player_x, player_y = self.player_position
                    return self.path_to_position((player_x, player_y), (x, y))
        
        # No clear direction found
        return ""
    
    def path_to_position(self, start_pos, end_pos) -> str:
        """
        Find a direction to move from start to end position.
        
        Args:
            start_pos: (x, y) or description of starting position
            end_pos: (x, y) or description of ending position
            
        Returns:
            str: Direction ("up", "down", "left", "right")
        """
        # Convert text positions to coordinates if needed
        if isinstance(start_pos, str):
            # Default to center if we can't parse
            start_x, start_y = 0, 0
        else:
            try:
                start_x, start_y = start_pos
            except (ValueError, TypeError):
                start_x, start_y = 0, 0
        
        if isinstance(end_pos, str):
            # Try to parse direction from text
            if "up" in end_pos.lower():
                return "up"
            elif "down" in end_pos.lower():
                return "down"
            elif "left" in end_pos.lower():
                return "left"
            elif "right" in end_pos.lower():
                return "right"
            else:
                # Can't determine, return empty
                return ""
        else:
            try:
                end_x, end_y = end_pos
            except (ValueError, TypeError):
                return ""
            
            # Simple greedy pathfinding - move in the direction of the largest difference
            dx = end_x - start_x
            dy = end_y - start_y
            
            if abs(dx) > abs(dy):
                # Move horizontally
                return "right" if dx > 0 else "left"
            else:
                # Move vertically
                return "down" if dy > 0 else "up"
    
    def execute_move(self, move: str) -> None:
        """
        Execute a move by pressing the corresponding button.
        
        Args:
            move: Direction or button to press
        """
        direction_mapping = {
            "up": Button.UP,
            "down": Button.DOWN,
            "left": Button.LEFT,
            "right": Button.RIGHT,
            "a": Button.A,
            "b": Button.B,
            "start": Button.START,
            "select": Button.SELECT
        }
        
        button = direction_mapping.get(move.lower())
        if button:
            self.controller.press_button(button)
        else:
            logger.warning(f"Unknown move: {move}")
    
    def execute_text_move(self, move_text: str) -> None:
        """
        Execute a move described in text.
        
        Args:
            move_text: Text description of the move
        """
        # Check for direction keywords
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
        
        move_text = move_text.lower()
        for keyword, button in direction_keywords.items():
            if keyword in move_text:
                self.controller.press_button(button)
                return
        
        logger.warning(f"Couldn't parse move text: {move_text}")
    
    def solve_ice_puzzle(self, exit_description: str = "exit") -> bool:
        """
        Specifically solve an ice puzzle using a hybrid of memory reading and vision.
        
        Args:
            exit_description: Description of the target exit
            
        Returns:
            bool: True if puzzle was solved
        """
        logger.info("Attempting to solve ice puzzle")
        
        # First, build a tile map of the ice puzzle
        tile_map = self.build_tile_map()
        
        # Then capture and analyze the screen for additional context
        self.capture_screen()
        analysis = self.analyze_screen()
        
        # Confirm we're actually on an ice puzzle
        is_ice_puzzle = False
        
        # Check the tile map for ice tiles
        if tile_map:
            for row in tile_map:
                if TileType.ICE in row:
                    is_ice_puzzle = True
                    break
        
        # Also check vision analysis
        if not is_ice_puzzle and "special_tiles" in analysis:
            for tile in analysis["special_tiles"]:
                if "ice" in str(tile).lower():
                    is_ice_puzzle = True
                    break
        
        if not is_ice_puzzle:
            logger.warning("No ice tiles detected in current view")
            return False
        
        # Determine the next move for the ice puzzle
        next_move = self.determine_ice_puzzle_move(analysis, exit_description)
        
        if next_move:
            logger.info(f"Ice puzzle move: {next_move}")
            self.execute_move(next_move)
            time.sleep(2)  # Wait longer for sliding to complete
            
            # Check if we reached the exit
            self.capture_screen()
            new_analysis = self.analyze_screen()
            
            if exit_description.lower() in str(new_analysis).lower():
                logger.info("Successfully exited the ice puzzle!")
                return True
            
            # Recursively continue solving the puzzle
            return self.solve_ice_puzzle(exit_description)
        
        logger.warning("Couldn't determine move for ice puzzle")
        return False
    
    def determine_ice_puzzle_move(self, analysis: Dict[str, Any], target: str) -> str:
        """
        Determine the best move for an ice puzzle.
        
        Args:
            analysis: Screen analysis from vision
            target: Target description
            
        Returns:
            str: Direction to move ("up", "down", "left", "right")
        """
        # Try to use memory and tile map first
        if self.tile_map:
            player_x, player_y = self.player_position
            
            # Try each direction and see where we'd end up
            directions = ["up", "down", "left", "right"]
            for direction in directions:
                end_x, end_y = self.predict_ice_slide_endpoint(player_x, player_y, direction)
                
                # Check if this endpoint gets us closer to the goal
                if "exits" in analysis:
                    for exit_info in analysis["exits"]:
                        if target.lower() in str(exit_info).lower() and "position" in exit_info:
                            exit_x, exit_y = exit_info["position"]
                            
                            # Calculate distances
                            current_dist = abs(player_x - exit_x) + abs(player_y - exit_y)
                            new_dist = abs(end_x - exit_x) + abs(end_y - exit_y)
                            
                            if new_dist < current_dist:
                                return direction
        
        # If memory approach didn't yield results, fall back to vision
        ice_puzzle_prompt = """
        This is a Pokémon ice puzzle where the player slides until hitting a wall or obstacle.
        1. Identify all ice tiles, walls, and obstacles
        2. Determine the current player position
        3. Calculate which direction to press to reach the exit or make progress
        4. Consider the sliding mechanics - player will slide until hitting something
        
        Provide a specific directional instruction (UP, DOWN, LEFT, RIGHT) that will help make progress.
        """
        
        img = Image.open(self.last_screenshot_path)
        response = self.model.generate_content([ice_puzzle_prompt, img])
        
        # Extract the recommended direction
        direction_text = response.text.strip().lower()
        
        for direction in ["up", "down", "left", "right"]:
            if direction in direction_text:
                return direction
        
        return ""
    
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
        Follow a described route using a hybrid of memory reading and vision.
        
        Args:
            route_description: Text description of the route to follow
            max_steps: Maximum number of steps to attempt
            
        Returns:
            bool: True if reached the destination
        """
        logger.info(f"Following route: {route_description}")
        
        for step in range(max_steps):
            # First, try to use memory to determine our position and the map
            map_data = self.read_map_from_memory()
            
            # Build a tile map (will use vision as fallback if memory reading fails)
            tile_map = self.build_tile_map()
            
            # Use vision for additional context and goal-oriented navigation
            prompt = f"""
            Help navigate through this Pokémon game to follow this route:
            "{route_description}"
            
            Observe the current state, determine my position, and provide the NEXT SINGLE ACTION
            I should take to make progress along this route.
            
            Just respond with one of: UP, DOWN, LEFT, RIGHT, A, B
            """
            
            # Capture screen if we don't have one yet
            if not self.last_screenshot_path or not os.path.exists(self.last_screenshot_path):
                self.capture_screen()
            
            # Ask vision API for next move
            img = Image.open(self.last_screenshot_path)
            response = self.model.generate_content([prompt, img])
            
            # Process the guidance
            action = response.text.strip().upper()
            logger.info(f"Step {step+1}/{max_steps}: Vision suggests: {action}")
            
            # Execute the suggested action
            self.execute_text_move(action)
            
            # Wait for the action to complete
            time.sleep(1)
            
            # Periodically check if we've reached the destination
            if step % 5 == 0 or "DESTINATION" in action or "ARRIVED" in action:
                self.capture_screen()
                status_prompt = f"Have I reached the destination of {route_description}? Answer YES or NO only."
                img = Image.open(self.last_screenshot_path)
                status_response = self.model.generate_content([status_prompt, img])
                
                if "YES" in status_response.text.upper():
                    logger.info(f"Destination reached: {route_description}")
                    return True
        
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