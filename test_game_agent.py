#!/usr/bin/env python3
"""
Test script for the GameAgent.

This script initializes a GameAgent with the Gemini provider
and runs it for a few steps to demonstrate the thinking and playing loop.
"""

import os
import argparse
from dotenv import load_dotenv
from game_agent import GameAgent

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test the GameAgent")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], 
                      default="gemini", help="LLM provider to use")
    parser.add_argument("--api-key", help="API key for the LLM provider (overrides .env file)")
    parser.add_argument("--steps", type=int, default=3, 
                      help="Number of steps to run")
    parser.add_argument("--delay", type=float, default=2.0,
                      help="Delay between steps in seconds")
    return parser.parse_args()

def main():
    """Main function to test the GameAgent."""
    # Parse arguments
    args = parse_arguments()
    
    # Load API keys from .env file
    load_dotenv()
    
    # Get API key from arguments or environment variable
    api_key = args.api_key
    if not api_key:
        if args.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
        elif args.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif args.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print(f"ERROR: No API key found for {args.provider}. Please provide it as an argument or in a .env file.")
        return 1
    
    # Create the GameAgent
    print(f"Initializing GameAgent with {args.provider} provider...")
    agent = GameAgent(
        llm_provider=args.provider,
        api_key=api_key,
        screenshot_dir="test_screenshots"
    )
    
    # Run the loop for the specified number of steps
    print(f"Running GameAgent for {args.steps} steps with {args.delay}s delay...")
    agent.run_loop(steps=args.steps, delay=args.delay)
    
    print("Test completed!")
    return 0

if __name__ == "__main__":
    exit(main()) 