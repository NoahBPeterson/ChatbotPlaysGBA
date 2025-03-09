"""
Base LLM Provider class that defines the interface for all providers.
"""

from typing import Dict, List, Optional


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, api_key: str):
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for the LLM provider
        """
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