#!/usr/bin/env python3
"""
Example Movement Script

This script demonstrates how to use the MGBAController to move a character around in a game.
It creates a simple pattern of movements using the directional buttons.
"""

import time
import random
from mgba_controller import MGBAController, Button

def move_in_pattern():
    """Move the character in a square pattern."""
    print("Moving in a square pattern...")
    
    # Initialize the controller
    controller = MGBAController()
    
    # Create a square movement pattern
    # Each tuple is (button, number of presses)
    pattern = [
        (Button.RIGHT, 5),
        (Button.DOWN, 5),
        (Button.LEFT, 5),
        (Button.UP, 5)
    ]
    
    # Execute the pattern
    for button, count in pattern:
        print(f"Moving {button.value} x{count}")
        for _ in range(count):
            controller.press_button(button, hold_duration_ms=100)
            time.sleep(0.2)  # Small delay between presses
        time.sleep(0.5)  # Pause at each corner
    
    print("Square pattern completed!")

def move_randomly():
    """Move the character in a random pattern."""
    print("Moving in a random pattern...")
    
    # Initialize the controller
    controller = MGBAController()
    
    # Available movement directions
    directions = [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT]
    
    # Make 20 random movements
    for i in range(20):
        button = random.choice(directions)
        print(f"Movement {i+1}: {button.value}")
        controller.press_button(button, hold_duration_ms=100)
        time.sleep(0.2)  # Small delay between presses
    
    print("Random movement completed!")

def main():
    """Main function to run the example."""
    print("mGBA Controller Movement Example")
    print("================================")
    
    try:
        # Ask user what pattern they want to run
        print("\nSelect a movement pattern:")
        print("1. Square pattern")
        print("2. Random movement")
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == "1":
            move_in_pattern()
        elif choice == "2":
            move_randomly()
        else:
            print("Invalid choice. Please enter 1 or 2.")
            return 1
            
    except Exception as e:
        print(f"Error during execution: {e}")
        return 1
        
    print("\nDone!")
    return 0

if __name__ == "__main__":
    main() 