#!/usr/bin/env python3
"""
Test script for taking a screenshot with the mGBA-http API and analyzing it with Gemini Vision.
"""

import os
import argparse
import time
from PIL import Image
from dotenv import load_dotenv
from mgba_controller import MGBAController
from vision_controller import VisionController

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test mGBA screenshot API and Gemini Vision analysis")
    parser.add_argument("--api-key", help="Gemini API key (overrides .env file)")
    parser.add_argument("--output", default="screenshot.png", help="Output file path for the screenshot")
    parser.add_argument("--vision-only", action="store_true", help="Only test vision analysis")
    return parser.parse_args()

def main():
    """Main function to test screenshot capture and vision analysis."""
    args = parse_arguments()
    
    # Load API key from .env file if not provided as argument
    load_dotenv()
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Gemini API key not found. Please provide it as an argument or in a .env file.")
        return 1
    
    print("Initializing controllers...")
    mgba_controller = MGBAController()
    vision_controller = VisionController(api_key=api_key, mgba_controller=mgba_controller)
    
    # Display game information
    print(f"Game Title: {mgba_controller.game_title}")
    print(f"Game Code: {mgba_controller.game_code}")
    
    # Capture screenshot if not in vision-only mode
    if not args.vision_only:
        print(f"Capturing screenshot to {args.output}...")
        start_time = time.time()
        screenshot_path = vision_controller.capture_screen(args.output)
        capture_time = time.time() - start_time
        
        # Verify the screenshot file exists and has content
        if os.path.exists(screenshot_path):
            file_size = os.path.getsize(screenshot_path) / 1024  # Size in KB
            print(f"Screenshot captured to: {screenshot_path}")
            print(f"Capture time: {capture_time:.2f} seconds")
            print(f"File size: {file_size:.2f} KB")
            
            # Open the image to verify it's valid
            img = Image.open(screenshot_path)
            print(f"Image dimensions: {img.size[0]}x{img.size[1]}")
            print("Screenshot captured successfully!")
        else:
            print(f"ERROR: Screenshot file {args.output} was not created!")
            return 1
    else:
        screenshot_path = args.output
        if not os.path.exists(screenshot_path):
            print(f"ERROR: Screenshot file {screenshot_path} not found for vision analysis!")
            return 1
    
    # Test vision analysis
    print("\nAnalyzing screenshot with Gemini Vision API...")
    start_time = time.time()
    analysis = vision_controller.analyze_screen(screenshot_path)
    analysis_time = time.time() - start_time
    
    print(f"Analysis time: {analysis_time:.2f} seconds")
    print("\nAnalysis Results:")
    
    # Display key parts of the analysis
    if "location" in analysis:
        print(f"Location: {analysis['location']}")
    if "description" in analysis:
        print(f"Description: {analysis['description']}")
    if "objects" in analysis:
        print(f"Objects detected: {len(analysis['objects'])}")
        for obj in analysis['objects'][:5]:  # Show first 5 objects
            print(f"  - {obj}")
    if "recommended_moves" in analysis:
        print("Recommended moves:")
        for move in analysis['recommended_moves'][:3]:  # Show top 3 moves
            print(f"  - {move}")
    
    print("\nComplete analysis result:")
    print(analysis)
    
    return 0

if __name__ == "__main__":
    exit(main()) 