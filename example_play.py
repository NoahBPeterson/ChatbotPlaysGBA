#!/usr/bin/env python3
"""
Example Play Script for Claude Plays Pokémon

This script demonstrates how to use the MGBAController to perform a simple sequence
of actions in a Pokémon game.
"""

import time
import random
import sys
from mgba_controller import MGBAController, Button

def random_walk(controller, steps=10, pause_between=0.2):
    """
    Perform a random walk in the game by pressing random directional buttons.
    
    Args:
        controller: The MGBAController instance
        steps: Number of steps to take
        pause_between: Pause between button presses in seconds
    """
    directions = [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT]
    
    print(f"\nPerforming random walk ({steps} steps)...")
    
    for i in range(steps):
        direction = random.choice(directions)
        print(f"Step {i+1}/{steps}: Moving {direction.name}")
        
        try:
            controller.press_button(direction)
            time.sleep(pause_between)  # Wait a bit between moves
        except Exception as e:
            print(f"Error during random walk: {e}")
            return False
    
    return True

def start_new_game(controller):
    """
    Simulate starting a new game by navigating the main menu.
    
    Args:
        controller: The MGBAController instance
    """
    print("\nNavigating to New Game...")
    
    # Assuming we're at the title screen, press START to continue
    print("Pressing START at title screen")
    controller.press_button(Button.START)
    time.sleep(1.5)
    
    # Navigate to "New Game" (assuming it's the first option)
    print("Selecting New Game")
    controller.press_button(Button.A)
    time.sleep(2)
    
    # Continue through intro text (assuming there's some dialog)
    for i in range(5):
        print(f"Advancing dialog {i+1}/5")
        controller.press_button(Button.A)
        time.sleep(0.8)

def battle_sequence(controller):
    """
    Simulate a simple Pokémon battle sequence.
    
    Args:
        controller: The MGBAController instance
    """
    print("\nSimulating battle sequence...")
    
    # Select "FIGHT" option
    print("Selecting FIGHT")
    controller.press_button(Button.A)
    time.sleep(1)
    
    # Select the first move
    print("Selecting first move")
    controller.press_button(Button.A)
    time.sleep(2)
    
    # Advance through battle text
    for i in range(3):
        print(f"Advancing battle text {i+1}/3")
        controller.press_button(Button.A)
        time.sleep(1)

def explore_area(controller):
    """
    Explore the current area with a mix of movements and interactions.
    
    Args:
        controller: The MGBAController instance
    """
    print("\nExploring area...")
    
    # Move around randomly
    random_walk(controller, steps=5)
    
    # Interact with something (press A)
    print("Interacting with object/NPC")
    controller.press_button(Button.A)
    time.sleep(1)
    
    # Advance through any dialog
    for i in range(3):
        print(f"Advancing dialog {i+1}/3")
        controller.press_button(Button.A)
        time.sleep(0.8)
    
    # Move a bit more
    random_walk(controller, steps=3)
    
    # Open menu
    print("Opening menu")
    controller.press_button(Button.START)
    time.sleep(1)
    
    # Close menu
    print("Closing menu")
    controller.press_button(Button.B)
    time.sleep(1)

def main():
    """Main entry point for the example play script."""
    
    controller = None
    
    try:
        print("=== Claude Plays Pokémon Example ===")
        print("Initializing controller...")
        
        # Initialize the controller with a timeout of 3 seconds
        controller = MGBAController(timeout=3)
        
        # Get the ROM title to confirm connection
        try:
            rom_title = controller.get_rom_title()
            print(f"Connected! ROM Title: {rom_title}")
        except Exception as e:
            print(f"Error getting ROM title: {e}")
            print("Make sure mGBA-http is running and a ROM is loaded.")
            return 1
        
        # Ask the user what demo they want to run
        print("\nSelect a demo:")
        print("1. Random Walk")
        print("2. Start New Game Sequence")
        print("3. Battle Sequence")
        print("4. Area Exploration")
        print("5. Run All")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            random_walk(controller, steps=15)
        elif choice == "2":
            start_new_game(controller)
        elif choice == "3":
            battle_sequence(controller)
        elif choice == "4":
            explore_area(controller)
        elif choice == "5":
            print("\n=== Running all demos ===")
            # Run through all demo sequences
            start_new_game(controller)
            time.sleep(1)
            random_walk(controller, steps=10)
            time.sleep(1)
            battle_sequence(controller)
            time.sleep(1)
            explore_area(controller)
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
            return 1
        
        print("\nDemo completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 