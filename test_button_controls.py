#!/usr/bin/env python3
"""
Test Button Controls

A simple script to test button controls in the MGBAController class.
This script demonstrates how to press buttons and execute sequences of button presses.
"""

import time
from mgba_controller import MGBAController, Button

def main():
    """Main function to test button controls."""
    print("Testing MGBAController button controls...")
    
    try:
        # Initialize the controller
        controller = MGBAController()
        
        # Log the game we're controlling
        print(f"Connected to {controller.game_title} ({controller.game_code})")
        
        # Test a single button press
        print("\nTesting single button press (UP)...")
        controller.press_button(Button.UP)
        time.sleep(0.5)
        
        # Test a sequence of button presses
        print("\nTesting button sequence (UP, DOWN, LEFT, RIGHT)...")
        controller.press_sequence([
            Button.UP, 
            Button.DOWN, 
            Button.LEFT, 
            Button.RIGHT
        ], hold_duration_ms=100, delay_between_ms=300)
        time.sleep(0.5)
        
        # Test pressing multiple buttons simultaneously
        print("\nTesting simultaneous button press (L + R)...")
        controller.press_buttons_simultaneously([Button.L, Button.R])
        time.sleep(0.5)
        
        # Test a higher-level action
        print("\nTesting 'open_menu' action...")
        controller.execute_action("open_menu")
        time.sleep(1)
        
        print("\nTesting 'close_menu' action...")
        controller.execute_action("close_menu")
        
        print("\nButton control tests completed successfully!")
        
    except Exception as e:
        print(f"Error during button control tests: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 