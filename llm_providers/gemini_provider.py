"""
Gemini provider implementation for the GameAgent.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional

from .base import LLMProvider

# Configure logging
logger = logging.getLogger('game_agent.gemini_provider')


class GeminiProvider(LLMProvider):
    """Google Gemini API provider for LLM interactions"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro-latest"):
        super().__init__(api_key)
        self.model = model
        
        # Import here to avoid dependency if not using Gemini
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Store imports for later use
            self.genai = genai
            
            # Create generation config
            self.generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 0,
                "max_output_tokens": 8192,
            }
            logger.info(f"Initialized Gemini provider with model: {model}")
        except ImportError:
            logger.warning("Google Generative AI package not installed. Please install it with 'pip install google-generativeai'")
    
    def generate_response(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """Generate a response using Google Gemini API with text generation"""
        try:
            logger.debug(f"Starting Gemini generation with prompt length: {len(prompt)}")
            
            # Prepare the conversation history
            history = []
            
            # Process previous conversation history
            for msg in context:
                if msg["role"] == "user":
                    history.append({"role": "user", "parts": [msg["content"]]})
                    logger.debug(f"Added user message to history with length: {len(msg['content'])}")
                elif msg["role"] == "assistant":
                    # Parse assistant messages to extract potential tool results
                    if "\n\n## Command Results\n" in msg["content"]:
                        content_parts = msg["content"].split("\n\n## Command Results\n", 1)
                        text_content = content_parts[0]
                        history.append({"role": "model", "parts": [text_content]})
                        logger.debug(f"Added assistant message to history with length: {len(text_content)}")
                    else:
                        history.append({"role": "model", "parts": [msg["content"]]})
                        logger.debug(f"Added assistant message to history with length: {len(msg['content'])}")
            
            logger.debug(f"Processed history with {len(history)} messages")
            
            # Create a system message to instruct the model on commands format
            system_instruction = """
            You are an AI assistant playing a Pokémon game. 
            Analyze the screen description and game state to determine the best actions.
            
            Your responses should follow a specific format:
            1. First, provide a brief analysis of the current situation
            2. Then, issue ONE of the following commands using the exact syntax:
            
            - press_button:BUTTON - Press a single button (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R)
            - press_sequence:BUTTON1,BUTTON2,BUTTON3 - Press a sequence of buttons in order
            - hold_button:BUTTON - Hold a button down continuously
            - release_button:BUTTON - Release a button that was being held down
            - wait:SECONDS - Wait for the specified number of seconds
            - read_memory:ADDRESS,SIZE,DOMAIN - Read from the game's memory
            - navigate_to:LOCATION - Navigate to a specific location
            - solve_puzzle:TYPE,TARGET - Solve a puzzle to reach a target
            
            Example response:
            "I see we're in the Pokémon Center. Let's heal our Pokémon.
            press_button:A"
            """
            
            # Create a chat session
            model = self.genai.GenerativeModel(
                model_name=self.model,
                generation_config=self.generation_config,
            )
            
            logger.debug(f"Created model: {self.model}")
            
            try:
                # Create chat session with history
                chat = model.start_chat(history=history) if history else model.start_chat()
                logger.debug("Successfully created chat session")
                
                # Send message with the updated prompt containing our system instruction
                logger.debug("Sending message with command format instruction")
                full_prompt = f"{system_instruction}\n\nCurrent game state:\n{prompt}"
                response = chat.send_message(content=full_prompt)
                logger.debug(f"Received response from Gemini API")
                
                # Process the response text to extract commands
                result_content = ""
                if hasattr(response, 'text') and response.text:
                    result_content = response.text
                    logger.debug(f"Received text response with length: {len(result_content)}")
                
                # Look for command patterns in the response
                command_patterns = [
                    r'press_button:([A-Z]+)',
                    r'press_sequence:([A-Z,]+)',
                    r'hold_button:([A-Z]+)',
                    r'release_button:([A-Z]+)',
                    r'wait:(\d+\.?\d*)',
                    r'read_memory:([^,\s]+),(\d+)(?:,([a-z]+))?',
                    r'navigate_to:([^,\n]+)',
                    r'solve_puzzle:([^,\n]+),([^,\n]+)'
                ]
                
                # Extract and format commands
                commands = []
                for pattern in command_patterns:
                    matches = re.findall(pattern, result_content)
                    if matches:
                        # Different patterns have different group structures
                        for match in matches:
                            if isinstance(match, tuple):
                                # Patterns with multiple capture groups
                                if len(match) == 3 and pattern.startswith(r'read_memory'):
                                    addr, size, domain = match
                                    domain_part = f",{domain}" if domain else ""
                                    commands.append(f"read_memory:{addr},{size}{domain_part}")
                                elif len(match) == 2 and pattern.startswith(r'solve_puzzle'):
                                    puzzle_type, target = match
                                    commands.append(f"solve_puzzle:{puzzle_type},{target}")
                                else:
                                    logger.warning(f"Unhandled multi-group match: {match} for pattern {pattern}")
                            else:
                                # Patterns with a single capture group
                                command_type = pattern.split(':')[0]
                                commands.append(f"{command_type}:{match}")
                
                # If specific commands were found, append them to the result
                if commands:
                    # Keep only the analysis part, remove any command text
                    for pattern in command_patterns:
                        result_content = re.sub(pattern, '', result_content)
                    
                    # Add the commands in a standard format
                    result_content = result_content.strip() + "\n\n" + "\n".join(commands)
                
                return result_content.strip()
                
            except Exception as e:
                logger.error(f"Error with chat session: {e}")
                return f"Error generating response: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return "Error generating response. Please check the logs." 