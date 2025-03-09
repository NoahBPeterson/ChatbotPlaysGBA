"""
OpenAI provider implementation for the GameAgent.
"""

import json
import logging
from typing import Dict, List

from .base import LLMProvider

# Configure logging
logger = logging.getLogger('game_agent.openai_provider')


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for LLM interactions"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        super().__init__(api_key)
        self.model = model
        
        # Import here to avoid dependency if not using OpenAI
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("OpenAI package not installed. Please install it with 'pip install openai'")
        
    def generate_response(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """Generate a response using OpenAI API with function calling"""
        try:
            # Define functions/tools for consistency with other providers
            functions = [
                {
                    "type": "function",
                    "function": {
                        "name": "press_button",
                        "description": "Press a game button momentarily (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "button": {
                                    "type": "string",
                                    "description": "The button to press (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)",
                                    "enum": ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]
                                }
                            },
                            "required": ["button"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "press_sequence",
                        "description": "Press a sequence of buttons in order, one after another",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "buttons": {
                                    "type": "string",
                                    "description": "Comma-separated list of buttons to press in sequence (e.g., 'UP,RIGHT,A')"
                                }
                            },
                            "required": ["buttons"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "hold_button",
                        "description": "Hold a button down continuously until released",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "button": {
                                    "type": "string",
                                    "description": "The button to hold down (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)",
                                    "enum": ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]
                                }
                            },
                            "required": ["button"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "release_button",
                        "description": "Release a button that was being held down",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "button": {
                                    "type": "string",
                                    "description": "The button to release (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)",
                                    "enum": ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]
                                }
                            },
                            "required": ["button"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "wait",
                        "description": "Wait for a specified number of seconds before taking another action",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "seconds": {
                                    "type": "number",
                                    "description": "Number of seconds to wait"
                                }
                            },
                            "required": ["seconds"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_memory",
                        "description": "Read a value from the game's memory at a specific address",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "address": {
                                    "type": "string",
                                    "description": "Memory address to read from (can be in hex format with 0x prefix)"
                                },
                                "size": {
                                    "type": "integer",
                                    "description": "Number of bytes to read (1, 2, 4, or more)"
                                },
                                "domain": {
                                    "type": "string",
                                    "description": "Memory domain to read from (wram, iwram, etc.), defaults to wram"
                                }
                            },
                            "required": ["address", "size", "domain"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "navigate_to",
                        "description": "Navigate to a specific location in the game using vision-based pathfinding",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "Location to navigate to (e.g., 'Pokemon Center', 'Route 1')"
                                }
                            },
                            "required": ["location"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "solve_puzzle",
                        "description": "Attempt to solve a puzzle of a specific type to reach a target",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "puzzle_type": {
                                    "type": "string",
                                    "description": "Type of puzzle to solve (e.g., 'ice', defaults to 'ice')"
                                },
                                "target": {
                                    "type": "string",
                                    "description": "Description of the target to reach (e.g., 'exit', 'ladder')"
                                }
                            },
                            "required": ["puzzle_type", "target"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            ]
            
            # Add the messages
            messages = []
            
            # Process previous conversation history
            for msg in context:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # System message to give context
            system_message = {
                "role": "system", 
                "content": "You are an AI assistant playing a Pokémon game. Analyze the game state and recommend actions using the provided functions."
            }
            
            # Add the system message at the beginning
            messages.insert(0, system_message)
            
            # Add the latest user message (prompt)
            messages.append({"role": "user", "content": prompt})
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=functions,
                tool_choice="auto",
                max_tokens=1000
            )
            
            # Process the response
            result_content = ""
            
            # Extract the assistant's message
            assistant_msg = response.choices[0].message
            
            # Add the thinking/reasoning if present
            if assistant_msg.content:
                result_content += assistant_msg.content + "\n\n"
            
            # Process any tool calls
            if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
                for tool_call in assistant_msg.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "press_button":
                        result_content += f"press_button:{function_args['button']}\n"
                    elif function_name == "press_sequence":
                        result_content += f"press_sequence:{function_args['buttons']}\n"
                    elif function_name == "hold_button":
                        result_content += f"hold_button:{function_args['button']}\n"
                    elif function_name == "release_button":
                        result_content += f"release_button:{function_args['button']}\n"
                    elif function_name == "wait":
                        result_content += f"wait:{function_args['seconds']}\n"
                    elif function_name == "read_memory":
                        domain_part = f",{function_args.get('domain', 'wram')}" if "domain" in function_args else ""
                        result_content += f"read_memory:{function_args['address']},{function_args['size']}{domain_part}\n"
                    elif function_name == "navigate_to":
                        result_content += f"navigate_to:{function_args['location']}\n"
                    elif function_name == "solve_puzzle":
                        puzzle_type = function_args.get("puzzle_type", "ice")
                        result_content += f"solve_puzzle:{puzzle_type},{function_args['target']}\n"
            
            return result_content.strip()
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return "Error generating response. Please check the logs." 