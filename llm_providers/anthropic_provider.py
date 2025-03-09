"""
Anthropic provider implementation for the GameAgent.
"""

import logging
from typing import Dict, List

from .base import LLMProvider

# Configure logging
logger = logging.getLogger('game_agent.anthropic_provider')


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for Claude interactions"""
    
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        super().__init__(api_key)
        self.model = model
        
        # Import here to avoid dependency if not using Anthropic
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.warning("Anthropic package not installed. Please install it with 'pip install anthropic'")
        
    def generate_response(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """Generate a response using Anthropic API with tool use integration"""
        try:
            # Convert context format to Anthropic's format
            messages = []
            
            # Process previous conversation history
            for msg in context:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    # Parse assistant messages to separate potential tool results
                    if "\n\n## Command Results\n" in msg["content"]:
                        content_parts = msg["content"].split("\n\n## Command Results\n", 1)
                        text_content = content_parts[0]
                        
                        # Add assistant's text response
                        messages.append({
                            "role": "assistant",
                            "content": text_content
                        })
                        
                        # If there were tool results, we'll add them to the history
                        if len(content_parts) > 1:
                            # We don't actually need to parse the results further here
                            # as they've already been processed in previous iterations
                            pass
                    else:
                        messages.append({"role": "assistant", "content": msg["content"]})
            
            # Define tools for the API call - these map to our available commands
            tools = [
                {
                    "name": "press_button",
                    "description": "Press a game button momentarily (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "button": {
                                "type": "string",
                                "description": "The button to press (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)",
                            }
                        },
                        "required": ["button"],
                    },
                },
                {
                    "name": "press_sequence",
                    "description": "Press a sequence of buttons in order, one after another",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "buttons": {
                                "type": "string",
                                "description": "Comma-separated list of buttons to press in sequence (e.g., 'UP,RIGHT,A')",
                            }
                        },
                        "required": ["buttons"],
                    },
                },
                {
                    "name": "hold_button",
                    "description": "Hold a button down continuously until released",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "button": {
                                "type": "string",
                                "description": "The button to hold down (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)",
                            }
                        },
                        "required": ["button"],
                    },
                },
                {
                    "name": "release_button",
                    "description": "Release a button that was being held down",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "button": {
                                "type": "string",
                                "description": "The button to release (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)",
                            }
                        },
                        "required": ["button"],
                    },
                },
                {
                    "name": "wait",
                    "description": "Wait for a specified number of seconds before taking another action",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "seconds": {
                                "type": "number",
                                "description": "Number of seconds to wait",
                            }
                        },
                        "required": ["seconds"],
                    },
                },
                {
                    "name": "read_memory",
                    "description": "Read a value from the game's memory at a specific address",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "address": {
                                "type": "string",
                                "description": "Memory address to read from (can be in hex format with 0x prefix)",
                            },
                            "size": {
                                "type": "integer",
                                "description": "Number of bytes to read (1, 2, 4, or more)",
                            },
                            "domain": {
                                "type": "string",
                                "description": "Memory domain to read from (wram, iwram, etc.), defaults to wram",
                            }
                        },
                        "required": ["address", "size"],
                    },
                },
                {
                    "name": "navigate_to",
                    "description": "Navigate to a specific location in the game using vision-based pathfinding",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to navigate to (e.g., 'Pokemon Center', 'Route 1')",
                            }
                        },
                        "required": ["location"],
                    },
                },
                {
                    "name": "solve_puzzle",
                    "description": "Attempt to solve a puzzle of a specific type to reach a target",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "puzzle_type": {
                                "type": "string",
                                "description": "Type of puzzle to solve (e.g., 'ice', defaults to 'ice')",
                            },
                            "target": {
                                "type": "string",
                                "description": "Description of the target to reach (e.g., 'exit', 'ladder')",
                            }
                        },
                        "required": ["target"],
                    },
                }
            ]
            
            # Add the latest user message (prompt)
            messages.append({"role": "user", "content": prompt})
            
            # Enable extended thinking for complex game states
            thinking_config = {
                "type": "enabled",
                "budget_tokens": 8000  # Allocate a reasonable token budget for thinking
            }
            
            # Make the API call with tools and thinking enabled
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                tools=tools,
                thinking=thinking_config,
                max_tokens=10000  # Increased from 4000 to be greater than thinking.budget_tokens (8000)
            )
            
            # Process the response to extract both thinking and tool use
            result_content = ""
            
            for content_block in response.content:
                if content_block.type == "text":
                    # Include all text content, which may include thinking
                    result_content += content_block.text + "\n"
                elif content_block.type == "tool_use":
                    # Convert tool_use blocks to our command format for compatibility
                    tool_name = content_block.name
                    tool_input = content_block.input
                    
                    # Format specific to each tool to match expected command format
                    if tool_name == "press_button":
                        result_content += f"press_button:{tool_input['button']}\n"
                    elif tool_name == "press_sequence":
                        result_content += f"press_sequence:{tool_input['buttons']}\n"
                    elif tool_name == "hold_button":
                        result_content += f"hold_button:{tool_input['button']}\n"
                    elif tool_name == "release_button":
                        result_content += f"release_button:{tool_input['button']}\n"
                    elif tool_name == "wait":
                        result_content += f"wait:{tool_input['seconds']}\n"
                    elif tool_name == "read_memory":
                        domain_part = f",{tool_input.get('domain', 'wram')}" if "domain" in tool_input else ""
                        result_content += f"read_memory:{tool_input['address']},{tool_input['size']}{domain_part}\n"
                    elif tool_name == "navigate_to":
                        result_content += f"navigate_to:{tool_input['location']}\n"
                    elif tool_name == "solve_puzzle":
                        puzzle_type = tool_input.get("puzzle_type", "ice")
                        result_content += f"solve_puzzle:{puzzle_type},{tool_input['target']}\n"
            
            return result_content.strip()
            
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            return "Error generating response. Please check the logs." 