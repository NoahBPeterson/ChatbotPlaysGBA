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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('game_agent')

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def generate_response(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            context: Previous conversation context
            
        Returns:
            str: The LLM's response
        """
        raise NotImplementedError("Subclasses must implement generate_response()")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for LLM interactions"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key)
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    def generate_response(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """Generate a response using OpenAI API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Add the new prompt to the conversation context
        messages = context.copy()
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return "Error generating response. Please check the logs."


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for Claude interactions"""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        super().__init__(api_key)
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        
    def generate_response(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """Generate a response using Anthropic API"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Convert the context format to Anthropic's format
        system_message = "You are an AI assistant playing a Pokémon game. Analyze the game state and recommend actions."
        
        # Extract just the content for the anthropic conversation history
        history = []
        for msg in context:
            if msg["role"] == "user":
                history.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                history.append({"role": "assistant", "content": msg["content"]})
        
        # Add the new prompt
        messages = history.copy()
        
        data = {
            "model": self.model,
            "system": system_message,
            "messages": messages + [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            return "Error generating response. Please check the logs."


class GeminiProvider(LLMProvider):
    """Google Gemini API provider for LLM interactions"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__(api_key)
        self.model = model
        
        # Import here to avoid dependency if not using Gemini
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.genai_model = genai.GenerativeModel(self.model)
        
    def generate_response(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """Generate a response using Google Gemini API"""
        try:
            # Convert context to Google's format
            chat = self.genai_model.start_chat(history=[])
            
            # Add previous messages to chat history
            for msg in context:
                if msg["role"] == "user":
                    chat.history.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    chat.history.append({"role": "model", "parts": [msg["content"]]})
            
            # Send the new prompt
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return "Error generating response. Please check the logs."


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
        mgba_controller: Optional[MGBAController] = None,
        vision_controller: Optional[VisionController] = None,
        history_length: int = 10,
        screenshot_dir: str = "agent_screenshots",
        session_log_file: Optional[str] = None
    ):
        """
        Initialize the GameAgent.
        
        Args:
            llm_provider: The LLM provider to use ("openai", "anthropic", or "gemini")
            api_key: API key for the LLM provider (or None to use environment variables)
            model: Model name to use with the provider (or None for default)
            mgba_controller: MGBAController instance (or None to create a new one)
            vision_controller: VisionController instance (or None to create a new one)
            history_length: Number of conversation exchanges to keep for context
            screenshot_dir: Directory to save screenshots in
            session_log_file: Path to log file (or None for default)
        """
        # Load API keys from environment if not provided
        load_dotenv()
        self.api_key = api_key or self._get_api_key_for_provider(llm_provider)
        
        # Initialize the LLM provider
        self.llm_provider_name = llm_provider.lower()
        self.llm = self._create_llm_provider(self.llm_provider_name, self.api_key, model)
        
        # Initialize controllers
        self.mgba_controller = mgba_controller or MGBAController()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.vision_controller = vision_controller or VisionController(
            api_key=gemini_api_key,
            mgba_controller=self.mgba_controller
        )
        
        # Set up conversation history
        self.history_length = history_length
        self.conversation_history = []
        
        # Set up screenshot directory
        self.screenshot_dir = screenshot_dir
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Set up session logging
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if session_log_file is None:
            os.makedirs("logs", exist_ok=True)
            self.session_log_file = f"logs/agent_session_{self.session_id}.json"
        else:
            self.session_log_file = session_log_file
            
        # Initialize session log
        self.session_log = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "game_title": self.mgba_controller.game_title,
            "game_code": self.mgba_controller.game_code,
            "llm_provider": llm_provider,
            "interactions": []
        }
        
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
        
        logger.info(f"GameAgent initialized for {self.mgba_controller.game_title} ({self.mgba_controller.game_code})")
        logger.info(f"Using LLM provider: {llm_provider}")
    
    def _get_api_key_for_provider(self, provider: str) -> str:
        """Get the API key for the specified provider from environment variables"""
        if provider.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider.lower() == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider.lower() == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
        if not api_key:
            raise ValueError(f"API key for {provider} not found in environment variables")
            
        return api_key
    
    def _create_llm_provider(self, provider: str, api_key: str, model: Optional[str]) -> LLMProvider:
        """Create the appropriate LLM provider instance"""
        if provider == "openai":
            return OpenAIProvider(api_key, model or "gpt-4o")
        elif provider == "anthropic":
            return AnthropicProvider(api_key, model or "claude-3-opus-20240229")
        elif provider == "gemini":
            return GeminiProvider(api_key, model or "gemini-1.5-pro")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _capture_game_state(self) -> Dict[str, Any]:
        """
        Capture the current game state from screenshots and memory.
        
        Returns:
            Dict with game state information
        """
        # Generate a timestamped filename for the screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_filename = f"{self.screenshot_dir}/screenshot_{timestamp}.png"
        
        # Capture screenshot
        screenshot_path = self.vision_controller.capture_screen(screenshot_filename)
        
        # Analyze the screenshot with vision
        vision_analysis = self.vision_controller.analyze_screen(screenshot_path)
        
        # Get memory-based game state
        memory_state = self.vision_controller.read_map_from_memory()
        
        # Try to build a tile map
        try:
            tile_map = self.vision_controller.build_tile_map()
        except Exception as e:
            logger.warning(f"Failed to build tile map: {e}")
            tile_map = []
        
        # Combine all the data into a game state dictionary
        game_state = {
            "screenshot_path": screenshot_path,
            "timestamp": timestamp,
            "vision_analysis": vision_analysis,
            "memory_state": memory_state,
            "tile_map": tile_map,
            "game_title": self.mgba_controller.game_title,
            "game_code": self.mgba_controller.game_code
        }
        
        return game_state
    
    def _format_prompt(self, game_state: Dict[str, Any]) -> str:
        """
        Format a prompt for the LLM based on the game state.
        
        Args:
            game_state: The current game state
            
        Returns:
            Formatted prompt string
        """
        prompt = [
            f"# Game State Analysis - {game_state['game_title']} ({game_state['game_code']})",
            f"Timestamp: {game_state['timestamp']}",
            "",
            "## Vision Analysis"
        ]
        
        vision = game_state.get("vision_analysis", {})
        if vision:
            prompt.extend([
                f"- Location: {vision.get('location', 'Unknown')}",
                f"- Player Position: {vision.get('player_position', 'Unknown')}"
            ])
            
            if "recommended_moves" in vision and vision["recommended_moves"]:
                prompt.append("- Suggested Moves:")
                for move in vision["recommended_moves"]:
                    prompt.append(f"  - {move}")
                    
            if "objects" in vision and vision["objects"]:
                prompt.append("- Objects:")
                for obj in vision["objects"][:5]:  # Limit to 5 objects to keep prompt concise
                    prompt.append(f"  - {obj}")
        
        prompt.extend(["", "## Memory State"])
        memory = game_state.get("memory_state", {})
        if memory:
            prompt.extend([
                f"- Map ID: {memory.get('map_id', 'Unknown')}",
                f"- Player Position (Memory): {memory.get('player_position', 'Unknown')}"
            ])
        
        prompt.extend([
            "",
            "## Available Commands",
            "- press_button: Press a button (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)",
            "- press_sequence: Press a sequence of buttons in order",
            "- hold_button: Hold a button down",
            "- release_button: Release a held button",
            "- wait: Wait for a specified number of seconds",
            "- read_memory: Read a value from memory",
            "- navigate_to: Navigate to a specific location",
            "- solve_puzzle: Try to solve a puzzle like an ice puzzle",
            "",
            "## Instructions",
            "1. Analyze the game state above",
            "2. Decide on the next action(s) to progress in the game",
            "3. Respond with one or more commands in EXACTLY this format: press_button:BUTTON",
            "   - Example: press_button:A or press_sequence:UP,RIGHT,A",
            "   - DO NOT add any extra formatting like asterisks or COMMAND: prefix",
            "4. Include your reasoning for these actions",
            "",
            "What should I do next?"
        ])
        
        return "\n".join(prompt)
    
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
        """Log the interaction to the session log file"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "game_state": {
                "location": game_state.get("vision_analysis", {}).get("location", "Unknown"),
                "map_id": game_state.get("memory_state", {}).get("map_id", "Unknown"),
                "screenshot_path": game_state.get("screenshot_path", "Unknown")
            },
            "prompt": prompt,
            "llm_response": response,
            "commands": commands,
            "results": results
        }
        
        self.session_log["interactions"].append(interaction)
        
        # Write the updated log to file
        with open(self.session_log_file, 'w') as f:
            json.dump(self.session_log, f, indent=2)
    
    def run_single_step(self) -> Dict[str, Any]:
        """
        Run a single step of the thinking and playing loop.
        
        Returns:
            Dict with details of the interaction
        """
        # Capture game state
        game_state = self._capture_game_state()
        
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
        Run the thinking and playing loop continuously or for a specific number of steps.
        
        Args:
            steps: Number of steps to run (0 for infinite loop)
            max_errors: Maximum number of consecutive errors before stopping
            delay: Delay between steps in seconds
        """
        step_count = 0
        error_count = 0
        
        try:
            while steps == 0 or step_count < steps:
                try:
                    logger.info(f"Running step {step_count + 1}" + ("" if steps == 0 else f"/{steps}"))
                    
                    # Run a single step
                    interaction = self.run_single_step()
                    
                    # Log key information
                    location = interaction["game_state"].get("vision_analysis", {}).get("location", "Unknown")
                    cmd_count = len(interaction["commands"])
                    logger.info(f"Location: {location} | Commands: {cmd_count}")
                    
                    # Reset error count on success
                    error_count = 0
                    
                    # Increment step count
                    step_count += 1
                    
                    # Delay before next step
                    if delay > 0:
                        time.sleep(delay)
                        
                except KeyboardInterrupt:
                    logger.info("Loop interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in step {step_count + 1}: {e}")
                    error_count += 1
                    
                    if error_count >= max_errors:
                        logger.error(f"Stopping loop after {max_errors} consecutive errors")
                        break
                        
                    # Delay after error
                    time.sleep(delay * 2)  # Longer delay after error
        finally:
            # Save final log
            self.session_log["end_time"] = datetime.now().isoformat()
            self.session_log["steps_completed"] = step_count
            with open(self.session_log_file, 'w') as f:
                json.dump(self.session_log, f, indent=2)
                
            logger.info(f"Session ended after {step_count} steps")
            logger.info(f"Session log saved to {self.session_log_file}")


def main():
    """Main function to run the GameAgent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run an AI agent to play Pokémon games")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default="openai",
                      help="LLM provider to use")
    parser.add_argument("--api-key", help="API key for the LLM provider")
    parser.add_argument("--model", help="Model name to use with the provider")
    parser.add_argument("--steps", type=int, default=0, 
                      help="Number of steps to run (0 for infinite)")
    parser.add_argument("--delay", type=float, default=1.0,
                      help="Delay between steps in seconds")
    parser.add_argument("--log-file", help="Path to log file")
    
    args = parser.parse_args()
    
    try:
        # Initialize the GameAgent
        agent = GameAgent(
            llm_provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            session_log_file=args.log_file
        )
        
        # Run the loop
        agent.run_loop(steps=args.steps, delay=args.delay)
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
