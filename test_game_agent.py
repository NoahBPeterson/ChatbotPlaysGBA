#!/usr/bin/env python3
"""
Test the GameAgent with different providers and configurations.
"""

import os
import argparse
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from game_agent import GameAgent

# Configure logging to include debug level for testing
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_game_agent')

# Set specific module loggers to debug
logging.getLogger('game_agent.gemini_provider').setLevel(logging.DEBUG)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the GameAgent")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini", "deepseek"],
                        help="LLM provider to use")
    parser.add_argument("--model", help="Model name to use with provider")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of steps to run (0 for infinite)")
    parser.add_argument("--delay", type=float, default=2.0, 
                        help="Delay between steps in seconds")
    parser.add_argument("--api-key", help="API key for the LLM provider")
    parser.add_argument("--adaptive-models", action="store_true", default=True,
                        help="Automatically switch to more powerful models when stuck (default: True)")
    parser.add_argument("--no-adaptive-models", action="store_false", dest="adaptive_models",
                        help="Disable automatic switching to more powerful models")
    
    return parser.parse_args()

def main():
    """Run the GameAgent test with specified provider and steps."""
    args = parse_arguments()
    
    # Create test screenshots directory
    os.makedirs("test_screenshots", exist_ok=True)
    
    # Get API key from arguments or environment
    api_key = args.api_key
    
    # Print initialization message
    print(f"Initializing GameAgent with {args.provider} provider...")
    if args.adaptive_models:
        print("Adaptive model selection is enabled - will switch to more powerful models if no progress is detected")
    else:
        print("Adaptive model selection is disabled - will use the specified model only")
    
    # Initialize and run the GameAgent
    agent = GameAgent(
        llm_provider=args.provider,
        api_key=api_key,
        model=args.model,
        screenshot_dir="test_screenshots",
        session_log_file=f"logs/agent_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        adaptive_models=args.adaptive_models
    )
    
    print(f"Running GameAgent for {args.steps} steps with {args.delay}s delay...")
    agent.run_loop(steps=args.steps, delay=args.delay)
    
    print("Test completed!")

if __name__ == "__main__":
    exit(main()) 