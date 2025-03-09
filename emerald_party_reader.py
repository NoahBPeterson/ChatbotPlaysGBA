#!/usr/bin/env python3
"""
Pokemon Emerald Party Reader

This script reads the player's Pokemon party from Pokemon Emerald via the mGBA-http API.
"""

import requests
import json
import sys
from mgba_controller import MGBAController

# The WRAM domain has a base address of 0x02000000
# The party count is at address 0x0244e9 relative to the WRAM base
# The party data starts at address 0x0244ec relative to the WRAM base
PARTY_COUNT_ADDR = 0x0244e9
PARTY_DATA_ADDR = 0x0244ec
PARTY_MON_SIZE = 100  # Each party member data structure is 100 bytes

def parse_pokemon_data(data):
    """Parse binary Pokemon data and return a dictionary of stats."""
    # For a full implementation, we would need to decode all the data structures
    # This is a simplified version that extracts a few key fields
    
    # First 4 bytes: Personality value (determines gender, nature, etc.)
    personality = int.from_bytes(data[0:4], byteorder='little')
    
    # Next 4 bytes: OT ID
    ot_id = int.from_bytes(data[4:8], byteorder='little')
    
    # At offset 80 we have the Pokemon's stats
    level = data[84]
    
    # Get HP values (current and max)
    hp = int.from_bytes(data[86:88], byteorder='little')
    max_hp = int.from_bytes(data[88:90], byteorder='little')
    
    # Get other stats
    attack = int.from_bytes(data[90:92], byteorder='little')
    defense = int.from_bytes(data[92:94], byteorder='little')
    speed = int.from_bytes(data[94:96], byteorder='little')
    
    return {
        'personality': personality,
        'ot_id': ot_id,
        'level': level,
        'hp': hp,
        'max_hp': max_hp,
        'attack': attack,
        'defense': defense,
        'speed': speed
    }

def main():
    """Main function to run the party reader."""
    print("Running Pokemon Emerald Party Reader...")
    
    try:
        # Initialize the MGBAController
        controller = MGBAController()
        
        # Check if this is actually Pokemon Emerald
        if not controller.game_title.startswith("POKEMON EMER"):
            print(f"Warning: This doesn't appear to be Pokemon Emerald. Detected game: {controller.game_title}")
            print("The script may not work correctly.")
        
        # Read the party count
        party_count = controller.read_byte(PARTY_COUNT_ADDR)
        print(f"Party count: {party_count}")
        
        if party_count == 0:
            print("No Pokemon in party.")
            return
        
        # Read each Pokemon in the party
        for i in range(party_count):
            # Calculate the address of this party member
            pokemon_addr = PARTY_DATA_ADDR + (i * PARTY_MON_SIZE)
            
            # Read the data
            data = controller.read_memory(pokemon_addr, PARTY_MON_SIZE)
            
            # Parse the data
            pokemon = parse_pokemon_data(data)
            
            # Print the Pokemon info
            print(f"\nPokemon {i+1}:")
            print(f"  Level: {pokemon['level']}")
            print(f"  HP: {pokemon['hp']}/{pokemon['max_hp']}")
            print(f"  Attack: {pokemon['attack']}")
            print(f"  Defense: {pokemon['defense']}")
            print(f"  Speed: {pokemon['speed']}")
            print(f"  Personality Value: 0x{pokemon['personality']:08x}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 