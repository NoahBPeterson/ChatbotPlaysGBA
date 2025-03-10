#!/usr/bin/env python3
"""
GameAgent - An AI agent that can play Pokémon games through LLM decision making.

This module provides a GameAgent class that:
1. Captures the game state (via screenshots and memory)
2. Sends this state to an LLM for analysis
3. Receives commands from the LLM
4. Executes the commands in the game
5. Repeats in a continuous thinking and playing loop
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dotenv import load_dotenv
import requests

from mgba_controller import MGBAController, Button
from vision_controller import VisionController

# Import providers from the new module
from llm_providers import LLMProvider, OpenAIProvider, AnthropicProvider, GeminiProvider, DeepSeekProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('game_agent')

class GameAgent:
    """
    AI agent that plays Pokémon games through LLM decision making.
    
    The agent continuously:
    1. Captures the game state
    2. Sends it to an LLM
    3. Executes commands from the LLM
    4. Repeats
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        vision_controller: Optional[VisionController] = None,
        mgba_controller: Optional[MGBAController] = None,
        history_length: int = 10,
        screenshot_dir: str = "agent_screenshots",
        session_log_file: Optional[str] = None,
        adaptive_models: bool = True
    ):
        """
        Initialize the GameAgent.
        
        Args:
            llm_provider: Provider to use ("openai", "anthropic", "gemini", "deepseek")
            api_key: API key for the LLM provider
            model: LLM model to use (or None for provider default)
            vision_controller: VisionController instance (or None to create a new one)
            mgba_controller: MGBAController instance (or None to create a new one)
            history_length: Number of conversation exchanges to keep for context
            screenshot_dir: Directory to save screenshots in
            session_log_file: Path to log file (or None for default)
            adaptive_models: Whether to automatically switch to more powerful models when no progress is detected
        """
        # Load environment variables if not provided
        if api_key is None:
            load_dotenv()
        
        # Set up the LLM provider
        self.llm_provider_name = llm_provider.lower()
        self.api_key = api_key or self._get_api_key_for_provider(self.llm_provider_name)
        
        # Save the original model choice for potential model switching
        self.original_model = model
        self.current_model = model
        self.adaptive_models = adaptive_models
        
        # Map of standard and fallback models for each provider
        self.model_tiers = {
            "openai": {
                "standard": "gpt-4o-mini",
                "fallback": "gpt-4o"
            },
            "anthropic": {
                "standard": "claude-3-haiku-20240307",
                "fallback": "claude-3-7-sonnet-20250219"
            },
            "gemini": {
                "standard": "gemini-1.5-flash",
                "fallback": "gemini-1.5-pro"
            },
            "deepseek": {
                "standard": "deepseek-chat",
                "fallback": "deepseek-chat"  # Currently only one model available
            }
        }
        
        # If no model was specified, use the standard model
        if self.current_model is None:
            self.current_model = self.model_tiers[self.llm_provider_name]["standard"]
        
        # Initialize the LLM provider with the current model
        self.llm = self._create_llm_provider(self.llm_provider_name, self.api_key, self.current_model)
        
        # Store the game title
        self.mgba_controller = mgba_controller or MGBAController()
        self.game_title = self.mgba_controller.game_title
        
        # Get provider-specific API key for vision
        vision_api_key = self._get_api_key_for_provider(self.llm_provider_name)
        
        # Initialize vision controller with the same provider when possible
        # Note: Some providers might not support vision, in which case we'll fall back to Gemini
        self.vision_controller = vision_controller or VisionController(
            api_key=vision_api_key,
            provider_name=self.llm_provider_name,  # Pass the same provider name
            mgba_controller=self.mgba_controller
        )
        
        # Dictionary of available commands the LLM can use
        self.available_commands = {
            "press_button": self._execute_press_button,
            "press_sequence": self._execute_press_sequence,
            "hold_button": self._execute_hold_button,
            "release_button": self._execute_release_button,
            "wait": self._execute_wait,
            "read_memory": self._execute_read_memory,
            "navigate_to": self._execute_navigate_to,
            "solve_puzzle": self._execute_solve_puzzle
        }
        
        # Session information
        self.conversation_history = []
        self.session_log = {
            "start_time": datetime.now().isoformat(),
            "game_title": self.game_title,
            "llm_provider": self.llm_provider_name,
            "interactions": []
        }
        self.log_file = None
        
        # Set up the conversation history with a maximum length
        self.history_length = history_length
        
        # Ensure the screenshot directory exists
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Set up logging
        if session_log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("logs", exist_ok=True)
            self.log_file = f"logs/agent_session_{timestamp}.json"
        else:
            self.log_file = session_log_file
        
        # Tracking game progress
        self.previous_positions = []
        self.no_progress_count = 0
        self.using_fallback_model = False
        
        logger.info(f"GameAgent initialized for {self.game_title}")
        logger.info(f"Using LLM provider: {self.llm_provider_name} with model: {self.current_model}")
        if self.adaptive_models:
            logger.info(f"Adaptive model selection is enabled (will switch to {self.model_tiers[self.llm_provider_name]['fallback']} if no progress detected)")
    
    def _get_api_key_for_provider(self, provider_name: str) -> str:
        """Get the appropriate API key for the specified provider."""
        if provider_name == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        elif provider_name == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "")
        elif provider_name == "gemini":
            return os.getenv("GEMINI_API_KEY", "")
        elif provider_name == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY", "")
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
    
    def _create_llm_provider(self, provider: str, api_key: str, model: Optional[str]) -> LLMProvider:
        """Create the appropriate LLM provider instance"""
        if provider == "openai":
            return OpenAIProvider(api_key, model or "gpt-4o-mini")
        elif provider == "anthropic":
            return AnthropicProvider(api_key, model or "claude-3-7-sonnet-20250219")
        elif provider == "gemini":
            return GeminiProvider(api_key, model or "gemini-1.5-pro-latest")
        elif provider == "deepseek":
            return DeepSeekProvider(api_key, model or "deepseek-chat")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _capture_game_state(self) -> Dict[str, Any]:
        """
        Capture the current game state using vision and memory.
        
        Returns:
            Dict: Game state information
        """
        state = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "game_title": self.game_title
        }
        
        # Capture and save a screenshot for this step
        screenshot_path = os.path.join(
            self.screenshot_dir,
            f"screenshot_{state['timestamp']}.png"
        )
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        screenshot_path = self.vision_controller.capture_screen(screenshot_path)
        
        # Get vision analysis
        analysis = self.vision_controller.analyze_screen(screenshot_path)
        state["analysis"] = analysis
        
        # Extract key information from analysis
        state["location"] = analysis.get("description", "")[:50] + "..." if analysis.get("description") else "Unknown"
        state["location_type"] = analysis.get("location_type", "unknown")
        state["player_position"] = analysis.get("player_position", (0, 0))
        state["map_id"] = analysis.get("map_id", 0)
        
        # Add memory information
        memory_data = self.vision_controller.read_map_from_memory()
        if memory_data:
            state["map_id"] = memory_data.get("map_id", state["map_id"])
            state["player_position_memory"] = memory_data.get("player_position", state["player_position"])
        
        return state
    
    def _format_prompt(self, game_state: Dict[str, Any]) -> str:
        """
        Format a prompt for the LLM based on the current game state.
        
        Args:
            game_state: Current game state
            
        Returns:
            Formatted prompt
        """
        location = game_state.get("location", "Unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get location type from analysis if available
        location_type = "unknown"
        if "analysis" in game_state and "location_type" in game_state["analysis"]:
            location_type = game_state["analysis"]["location_type"]
        
        # Check if this is likely a title screen based on various indicators
        is_title_screen = False
        if location_type == "title_screen":
            is_title_screen = True
        elif any(term in str(location).lower() for term in ["title screen", "main menu", "press start", "game freak"]):
            is_title_screen = True
            location_type = "title_screen"
        
        # Build the prompt
        prompt = f"# Game State Analysis - {self.game_title}\n"
        prompt += f"Timestamp: {timestamp}\n\n"
        
        # Special title screen banner if detected
        if is_title_screen:
            prompt += f"⭐️ TITLE SCREEN DETECTED: Press START to begin the game! ⭐️\n\n"
        
        # Vision analysis section
        prompt += f"## Vision Analysis\n"
        prompt += f"- Location: {location}\n"
        if "player_position" in game_state:
            prompt += f"- Player Position: {game_state['player_position']}\n"
        
        # Memory state section
        prompt += f"\n## Memory State\n"
        if "map_id" in game_state:
            prompt += f"- Map ID: {game_state['map_id']}\n"
        if "player_position_memory" in game_state:
            prompt += f"- Player Position (Memory): {game_state['player_position_memory']}\n"
        
        # Available commands section
        prompt += f"\n## Available Commands\n"
        prompt += f"- press_button: Press a button (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)\n"
        prompt += f"- press_sequence: Press a sequence of buttons in order\n"
        prompt += f"- hold_button: Hold a button down\n"
        prompt += f"- release_button: Release a held button\n"
        prompt += f"- wait: Wait for a specified number of seconds\n"
        prompt += f"- read_memory: Read a value from memory\n"
        prompt += f"- navigate_to: Navigate to a specific location\n"
        prompt += f"- solve_puzzle: Try to solve a puzzle like an ice puzzle\n"
        
        # Instructions section
        prompt += f"\n## Instructions\n"
        
        # Special instructions for title screen
        if is_title_screen:
            prompt += f"1. YOU ARE AT THE TITLE SCREEN. Press START to begin the game.\n"
            prompt += f"2. If START doesn't work, try pressing A instead.\n"
            prompt += f"3. The correct command format is: press_button:START\n\n"
        else:
            prompt += f"1. Analyze the game state above\n"
            prompt += f"2. Decide on the next action(s) to progress in the game\n"
            prompt += f"3. Respond with one or more commands in EXACTLY this format: press_button:BUTTON\n"
            prompt += f"   - Example: press_button:A or press_sequence:UP,RIGHT,A\n"
            prompt += f"   - DO NOT add any extra formatting like asterisks or COMMAND: prefix\n"
            prompt += f"4. Include your reasoning for these actions\n\n"
            
            # Add general title screen guidance
            prompt += f"Note: If you see a title screen or main menu, use press_button:START to begin the game.\n\n"
        
        prompt += f"What should I do next?"
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse the LLM's response into executable commands.
        
        Args:
            response: The LLM's response string
            
        Returns:
            List of command dictionaries with "command" and "params" keys
        """
        commands = []
        lines = response.strip().split("\n")
        
        for line in lines:
            # Remove any formatting characters like asterisks, etc.
            line = line.strip().replace("*", "").replace("**", "").replace("`", "")
            
            # Check for command prefix patterns
            if "COMMAND:" in line or "command:" in line:
                # Extract the actual command part
                cmd_part = line.split("COMMAND:", 1)[-1] if "COMMAND:" in line else line.split("command:", 1)[-1]
                
                # Handle commands with double colons (e.g., "press_button:RIGHT")
                if ":" in cmd_part:
                    parts = cmd_part.split(":", 1)
                    cmd = parts[0].strip().lower()
                    params = parts[1].strip()
                    
                    if cmd in self.available_commands:
                        commands.append({"command": cmd, "params": params})
            # Also check for direct command patterns without the COMMAND: prefix
            elif ":" in line and not line.startswith("http"):
                parts = line.split(":", 1)
                cmd = parts[0].strip().lower()
                params = parts[1].strip()
                
                if cmd in self.available_commands:
                    commands.append({"command": cmd, "params": params})
            
        return commands
    
    def _execute_commands(self, commands: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Execute a list of commands and return the results.
        
        Args:
            commands: List of command dictionaries
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for cmd_dict in commands:
            cmd = cmd_dict["command"]
            params = cmd_dict["params"]
            
            if cmd in self.available_commands:
                try:
                    result = self.available_commands[cmd](params)
                    results.append({"command": cmd, "params": params, "result": result, "success": True})
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error executing command {cmd}: {error_msg}")
                    results.append({
                        "command": cmd, 
                        "params": params, 
                        "result": f"Error: {error_msg}", 
                        "success": False
                    })
            else:
                results.append({
                    "command": cmd,
                    "params": params,
                    "result": f"Error: Unknown command '{cmd}'",
                    "success": False
                })
                
        return results
    
    def _format_results(self, results: List[Dict[str, str]]) -> str:
        """Format command results for the next LLM prompt"""
        lines = ["## Command Results"]
        
        for res in results:
            status = "✅" if res["success"] else "❌"
            lines.append(f"{status} {res['command']}: {res['params']}")
            lines.append(f"   Result: {res['result']}")
            lines.append("")
            
        return "\n".join(lines)
    
    def _execute_press_button(self, params: str) -> str:
        """Execute press_button command"""
        button_name = params.strip().upper()
        try:
            button = Button[button_name]
            self.mgba_controller.press_button(button)
            return f"Button {button_name} pressed successfully"
        except KeyError:
            raise ValueError(f"Invalid button name: {button_name}")
    
    def _execute_press_sequence(self, params: str) -> str:
        """Execute press_sequence command"""
        button_names = [b.strip().upper() for b in params.split(",")]
        buttons = []
        
        for name in button_names:
            try:
                button = Button[name]
                buttons.append(button)
            except KeyError:
                raise ValueError(f"Invalid button name: {name}")
                
        self.mgba_controller.press_sequence(buttons)
        return f"Button sequence {','.join(button_names)} pressed successfully"
    
    def _execute_hold_button(self, params: str) -> str:
        """Execute hold_button command"""
        button_name = params.strip().upper()
        try:
            button = Button[button_name]
            self.mgba_controller.hold_button(button)
            return f"Button {button_name} is now being held down"
        except KeyError:
            raise ValueError(f"Invalid button name: {button_name}")
    
    def _execute_release_button(self, params: str) -> str:
        """Execute release_button command"""
        button_name = params.strip().upper()
        try:
            button = Button[button_name]
            self.mgba_controller.release_button(button)
            return f"Button {button_name} released successfully"
        except KeyError:
            raise ValueError(f"Invalid button name: {button_name}")
    
    def _execute_wait(self, params: str) -> str:
        """Execute wait command"""
        try:
            seconds = float(params.strip())
            time.sleep(seconds)
            return f"Waited for {seconds} seconds"
        except ValueError:
            raise ValueError(f"Invalid wait time: {params}")
    
    def _execute_read_memory(self, params: str) -> str:
        """Execute read_memory command"""
        params_parts = params.strip().split(",")
        if len(params_parts) < 2:
            raise ValueError("read_memory requires address and size parameters")
            
        try:
            address = int(params_parts[0].strip(), 0)  # Using 0 as base allows for hex (0x...) format
            size = int(params_parts[1].strip())
            
            domain = "wram"  # Default to wram
            if len(params_parts) >= 3:
                domain = params_parts[2].strip()
                
            if size == 1:
                value = self.mgba_controller.read_byte(address, domain)
                return f"Value at {hex(address)} in {domain}: {value} (0x{value:02X})"
            elif size == 2:
                value = self.mgba_controller.read_short(address, domain)
                return f"Value at {hex(address)} in {domain}: {value} (0x{value:04X})"
            elif size == 4:
                value = self.mgba_controller.read_word(address, domain)
                return f"Value at {hex(address)} in {domain}: {value} (0x{value:08X})"
            else:
                bytes_data = self.mgba_controller.read_bytes(address, size, domain)
                hex_values = [f"0x{b:02X}" for b in bytes_data]
                return f"Bytes at {hex(address)} in {domain}: {', '.join(hex_values)}"
        except ValueError:
            raise ValueError(f"Invalid memory parameters: {params}")
        except Exception as e:
            raise ValueError(f"Error reading memory: {str(e)}")
    
    def _execute_navigate_to(self, params: str) -> str:
        """Execute navigate_to command"""
        target = params.strip()
        try:
            result = self.vision_controller.navigate_complex_area(target)
            if result:
                return f"Successfully navigated to {target}"
            else:
                return f"Failed to navigate to {target} within allowed steps"
        except Exception as e:
            raise ValueError(f"Navigation error: {str(e)}")
    
    def _execute_solve_puzzle(self, params: str) -> str:
        """Execute solve_puzzle command"""
        puzzle_type = "ice"  # Default to ice puzzle
        target = "exit"      # Default target
        
        params_parts = params.strip().split(",")
        if len(params_parts) >= 1:
            puzzle_type = params_parts[0].strip().lower()
            
        if len(params_parts) >= 2:
            target = params_parts[1].strip()
            
        if puzzle_type == "ice":
            try:
                result = self.vision_controller.solve_ice_puzzle(target)
                if result:
                    return f"Successfully solved ice puzzle to reach {target}"
                else:
                    return f"Failed to solve ice puzzle to reach {target}"
            except Exception as e:
                raise ValueError(f"Error solving ice puzzle: {str(e)}")
        else:
            raise ValueError(f"Unsupported puzzle type: {puzzle_type}")
    
    def _update_conversation_history(self, prompt: str, response: str, results: List[Dict[str, str]]) -> None:
        """Update the conversation history with the latest exchange"""
        # Format the results for inclusion in the next prompt
        results_str = self._format_results(results)
        
        # Add the prompt to history
        self.conversation_history.append({"role": "user", "content": prompt})
        
        # Add the response and results to history
        full_response = response
        if results:
            full_response += "\n\n" + results_str
            
        self.conversation_history.append({"role": "assistant", "content": full_response})
        
        # Limit history to the specified length
        while len(self.conversation_history) > self.history_length * 2:  # *2 because each exchange is 2 entries
            self.conversation_history.pop(0)
    
    def _log_interaction(self, game_state: Dict[str, Any], prompt: str, response: str, commands: List[Dict[str, str]], results: List[Dict[str, str]]) -> None:
        """
        Log an interaction to the session log.
        
        Args:
            game_state: Game state at the time of the interaction
            prompt: Prompt sent to the LLM
            response: Response from the LLM
            commands: Commands extracted from the response
            results: Results of executing the commands
        """
        # Initialize session_log if it's not a dictionary
        if not isinstance(self.session_log, dict):
            self.session_log = {
                "start_time": datetime.now().isoformat(),
                "game_title": self.game_title,
                "llm_provider": self.llm_provider_name,
                "interactions": []
            }
        
        # Ensure interactions list exists
        if "interactions" not in self.session_log:
            self.session_log["interactions"] = []
        
        # Create the interaction log
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "llm_response": response,
            "commands": commands,
            "results": results,
            "location": game_state.get("location", "Unknown"),
            "location_type": game_state.get("location_type", "unknown"),
            "map_id": game_state.get("map_id", 0)
        }
        
        # Add to session log
        self.session_log["interactions"].append(interaction)
        
        # Write the updated log to file
        with open(self.log_file, 'w') as f:
            json.dump(self.session_log, f, indent=2)
    
    def run_single_step(self) -> Dict[str, Any]:
        """
        Run a single step of the thinking and playing loop.
        
        Returns:
            Dict with details of the interaction
        """
        # Capture game state
        game_state = self._capture_game_state()
        
        # Track player position to detect when we're stuck
        if "player_position" in game_state:
            self._track_progress(game_state["player_position"])
        
        # Format prompt for LLM
        prompt = self._format_prompt(game_state)
        
        # Get response from LLM
        llm_response = self.llm.generate_response(prompt, self.conversation_history)
        
        # Parse commands from the response
        commands = self._parse_llm_response(llm_response)
        
        # Execute the commands
        results = self._execute_commands(commands)
        
        # Update conversation history
        self._update_conversation_history(prompt, llm_response, results)
        
        # Log the interaction
        self._log_interaction(game_state, prompt, llm_response, commands, results)
        
        # If we've recently switched models, log it in the results
        if self.using_fallback_model:
            model_info = {
                "command": "system_info", 
                "params": f"model_switch", 
                "result": f"Using more powerful model ({self.current_model}) for better reasoning",
                "success": True
            }
            results.append(model_info)
        
        # Return details of the interaction
        return {
            "game_state": game_state,
            "prompt": prompt,
            "llm_response": llm_response,
            "commands": commands,
            "results": results
        }
    
    def run_loop(self, steps: int = 0, max_errors: int = 3, delay: float = 1.0) -> None:
        """
        Run the game agent in a continuous loop.
        
        Args:
            steps: Number of steps to run (0 for infinite)
            max_errors: Maximum number of consecutive errors before stopping
            delay: Delay between steps in seconds
        """
        step_count = 0
        error_count = 0
        
        try:
            while steps == 0 or step_count < steps:
                logger.info(f"Running step {step_count+1}/{steps if steps > 0 else 'inf'}")
                
                try:
                    self.run_single_step()
                    step_count += 1
                    error_count = 0  # Reset error count after successful step
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error in step {step_count+1}: {e}")
                    
                    if error_count >= max_errors:
                        logger.error(f"Stopping loop after {error_count} consecutive errors")
                        break
                
                # Wait between steps
                if steps == 0 or step_count < steps:
                    time.sleep(delay)
        
        finally:
            # Make sure we always save the session log
            if isinstance(self.session_log, dict):
                self.session_log["end_time"] = datetime.now().isoformat()
                self.session_log["steps_completed"] = step_count
                with open(self.log_file, 'w') as f:
                    json.dump(self.session_log, f, indent=2)
                
            logger.info(f"Session ended after {step_count} steps")
            logger.info(f"Session log saved to {self.log_file}")

    def _switch_to_fallback_model(self):
        """
        Switch from the standard model to the more powerful fallback model.
        """
        if not self.adaptive_models or self.using_fallback_model:
            return False
        
        fallback_model = self.model_tiers[self.llm_provider_name]["fallback"]
        if self.current_model == fallback_model:
            return False
        
        logger.info(f"No progress detected for {self.no_progress_count} steps. Switching from {self.current_model} to {fallback_model}")
        
        # Create a new LLM provider with the fallback model
        self.current_model = fallback_model
        self.llm = self._create_llm_provider(self.llm_provider_name, self.api_key, self.current_model)
        self.using_fallback_model = True
        
        # Add a note to the conversation history
        self.conversation_history.append({
            "role": "system",
            "content": f"The agent has automatically switched to a more powerful model ({fallback_model}) because no progress was detected."
        })
        
        return True

    def _track_progress(self, current_position):
        """
        Track the player's position to detect when no progress is being made.
        
        Returns True if progress has been made, False otherwise.
        """
        # Add current position to history (limit to last 5 positions)
        self.previous_positions.append(current_position)
        if len(self.previous_positions) > 5:
            self.previous_positions.pop(0)
        
        # Not enough history to determine if we're stuck
        if len(self.previous_positions) < 3:
            return True
        
        # Check if we've been in the same position for the last 3 steps
        if all(pos == current_position for pos in self.previous_positions[-3:]):
            self.no_progress_count += 1
            logger.warning(f"No movement detected for {self.no_progress_count} consecutive checks")
            
            # If we've been stuck for 5 steps and we're not using the fallback model yet, switch models
            if self.no_progress_count >= 5 and not self.using_fallback_model and self.adaptive_models:
                self._switch_to_fallback_model()
            
            return False
        else:
            # Progress detected, reset counter
            if self.no_progress_count > 0:
                logger.info(f"Progress detected! Reset no-progress counter from {self.no_progress_count} to 0")
                self.no_progress_count = 0
            return True


def main():
    """Main function to run the GameAgent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run an AI agent to play Pokémon games")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini", "deepseek"], default="openai",
                       help="LLM provider to use")
    parser.add_argument("--api-key", help="API key for the LLM provider")
    parser.add_argument("--model", help="Model name to use with the provider")
    parser.add_argument("--steps", type=int, default=0, 
                       help="Number of steps to run (0 for infinite)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between steps in seconds")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("--adaptive-models", action="store_true", default=True,
                       help="Automatically switch to more powerful models when stuck (default: True)")
    parser.add_argument("--no-adaptive-models", action="store_false", dest="adaptive_models",
                       help="Disable automatic switching to more powerful models")
    
    # Add information about available models
    model_help = """
    Available models by provider:
      - openai: gpt-4o-mini (default, faster), gpt-4o (more capable)
      - anthropic: claude-3-haiku-20240307 (default), claude-3-7-sonnet-20250219
      - gemini: gemini-1.5-flash (faster), gemini-1.5-pro (more capable)
      - deepseek: deepseek-chat (default)
    
    Note: With adaptive models enabled (default), the agent will:
    1. Start with cheaper, faster models (if no specific model provided)
    2. Automatically switch to more powerful models if stuck in the same position
    """
    
    print(model_help)
    
    args = parser.parse_args()
    
    try:
        # Initialize the GameAgent
        agent = GameAgent(
            llm_provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            session_log_file=args.log_file,
            adaptive_models=args.adaptive_models
        )
        
        # Run the loop
        agent.run_loop(steps=args.steps, delay=args.delay)
        
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
