#!/usr/bin/env python3
"""
mGBA Controller Module

A high-level wrapper around the mGBA-http API for use in the Claude Plays Pokémon project.
This module provides easy-to-use functions for controlling the emulator and reading game state.
"""

import requests
import time
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mgba_controller')

class Button(Enum):
    """GameBoy button enumeration."""
    A = "A"
    B = "B"
    START = "Start"
    SELECT = "Select"
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    L = "L"  # For GBA
    R = "R"  # For GBA

class MemoryDomain(Enum):
    """mGBA memory domain enumeration."""
    WRAM = "wram"       # Main work RAM (0x02000000-0x0203FFFF)
    IWRAM = "iwram"     # Internal work RAM (0x03000000-0x03007FFF)
    BIOS = "bios"       # GBA BIOS
    CART0 = "cart0"     # Cartridge ROM, first bank
    CART1 = "cart1"     # Cartridge ROM, second bank
    CART2 = "cart2"     # Cartridge ROM, third bank
    SRAM = "sram"       # Save RAM
    VRAM = "vram"       # Video RAM
    OAM = "oam"         # Object Attribute Memory (sprites)
    PALETTE = "palette" # Color palettes
    IO = "io"           # I/O registers

class PokemonGame(Enum):
    """Pokemon game types."""
    RUBY = "POKEMON RUBY"
    SAPPHIRE = "POKEMON SAPP"
    EMERALD = "POKEMON EMER"
    FIRERED = "POKEMON FIRE"
    LEAFGREEN = "POKEMON LEAF"

class MGBAController:
    """
    High-level controller for mGBA emulator via the HTTP API.
    """
    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 5):
        """
        Initialize the controller with the mGBA-http server URL.
        
        Args:
            base_url: The base URL of the mGBA-http server
            timeout: Timeout in seconds for API calls
        """
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self._check_connection()
        
        # Identify the game
        self.game_title = self.get_game_title()
        self.game_code = self.get_game_code()
        logger.info(f"Game detected: {self.game_title} ({self.game_code})")
        
        # Set up memory domains
        self.memory_domains = self._get_memory_domains()
        
    def _check_connection(self) -> bool:
        """
        Check if the connection to the mGBA-http server is working.
        
        Returns:
            bool: True if connected, raises an exception otherwise
        """
        try:
            # Access the Swagger docs to check connectivity
            response = self.session.get(f"{self.base_url}/swagger/v0.4/swagger.json", timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Successfully connected to mGBA-http at {self.base_url}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to mGBA-http: {e}")
            raise ConnectionError(f"Could not connect to mGBA-http at {self.base_url}. "
                                 "Please ensure mGBA and mGBA-http are running properly.")

    def _get_memory_domains(self) -> Dict[str, Dict[str, int]]:
        """
        Get information about all available memory domains.
        
        Returns:
            Dict: Memory domain information (base, size, bound)
        """
        domains = {}
        
        for domain in MemoryDomain:
            try:
                base = self.get_memory_domain_base(domain.value)
                size = self.get_memory_domain_size(domain.value)
                bound = self.get_memory_domain_bound(domain.value)
                
                domains[domain.value] = {
                    'base': base,
                    'size': size,
                    'bound': bound
                }
                
                logger.info(f"Memory domain {domain.value}: base=0x{base:x}, size=0x{size:x}, bound=0x{bound:x}")
            except Exception as e:
                logger.warning(f"Failed to get information for memory domain {domain.value}: {e}")
        
        return domains

    def get_game_title(self) -> str:
        """
        Get the title of the currently loaded ROM.
        
        Returns:
            str: The ROM title
        """
        response = self.session.get(f"{self.base_url}/core/getGameTitle", timeout=self.timeout)
        response.raise_for_status()
        return response.text.strip()
    
    def get_game_code(self) -> str:
        """
        Get the game code of the currently loaded ROM.
        
        Returns:
            str: The game code (e.g., "AGB-BPEE" for Emerald)
        """
        response = self.session.get(f"{self.base_url}/core/getGameCode", timeout=self.timeout)
        response.raise_for_status()
        return response.text.strip()
    
    def get_memory_domain_base(self, domain: str) -> int:
        """
        Get the base address of a memory domain.
        
        Args:
            domain: Memory domain name
            
        Returns:
            int: Base address
        """
        response = self.session.get(
            f"{self.base_url}/memorydomain/base", 
            params={"memoryDomain": domain},
            timeout=self.timeout
        )
        response.raise_for_status()
        return int(response.text)
    
    def get_memory_domain_size(self, domain: str) -> int:
        """
        Get the size of a memory domain.
        
        Args:
            domain: Memory domain name
            
        Returns:
            int: Size in bytes
        """
        response = self.session.get(
            f"{self.base_url}/memorydomain/size", 
            params={"memoryDomain": domain},
            timeout=self.timeout
        )
        response.raise_for_status()
        return int(response.text)
    
    def get_memory_domain_bound(self, domain: str) -> int:
        """
        Get the upper bound address of a memory domain.
        
        Args:
            domain: Memory domain name
            
        Returns:
            int: Upper bound address
        """
        response = self.session.get(
            f"{self.base_url}/memorydomain/bound", 
            params={"memoryDomain": domain},
            timeout=self.timeout
        )
        response.raise_for_status()
        return int(response.text)

    def press_button(self, button: Union[Button, str], hold_duration_ms: int = 100) -> bool:
        """
        Press a single button.
        
        Args:
            button: The button to press (can be Button enum or string)
            hold_duration_ms: How long to hold the button in milliseconds
            
        Returns:
            bool: True if successful
        """
        if isinstance(button, Button):
            button = button.value
            
        # Use the tap endpoint (which handles both press and release)
        response = self.session.post(
            f"{self.base_url}/mgba-http/button/tap", 
            params={"key": button},
            timeout=self.timeout
        )
        response.raise_for_status()
        return True
    
    def press_buttons(self, buttons: List[Union[Button, str]], hold_duration_ms: int = 100) -> bool:
        """
        Press multiple buttons in sequence.
        
        Args:
            buttons: List of buttons to press (can be Button enums or strings)
            hold_duration_ms: How long to hold each button in milliseconds
            
        Returns:
            bool: True if successful
        """
        success = True
        for button in buttons:
            try:
                self.press_button(button, hold_duration_ms)
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to press button {button}: {e}")
                success = False
        
        return success
    
    def press_buttons_simultaneously(self, buttons: List[Union[Button, str]]) -> bool:
        """
        Press multiple buttons simultaneously.
        
        Args:
            buttons: List of buttons to press simultaneously (can be Button enums or strings)
            
        Returns:
            bool: True if successful
        """
        button_list = [button.value if isinstance(button, Button) else button for button in buttons]
        
        # It seems the API doesn't support multiple buttons with tapmany
        # Let's use individual taps for each button with a very short delay
        for button in button_list:
            try:
                # Use the tap endpoint for each button
                response = self.session.post(
                    f"{self.base_url}/mgba-http/button/tap", 
                    params={"key": button},
                    timeout=self.timeout
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to press button {button}: {e}")
                return False
                
        return True
    
    def press_sequence(self, sequence: List[Union[Button, str]], 
                      hold_duration_ms: int = 100, delay_between_ms: int = 50) -> bool:
        """
        Press a sequence of buttons with delays between each.
        
        Args:
            sequence: List of buttons to press in sequence
            hold_duration_ms: How long to hold each button in milliseconds
            delay_between_ms: Delay between button presses in milliseconds
            
        Returns:
            bool: True if all buttons were pressed successfully
        """
        success = True
        for button in sequence:
            try:
                self.press_button(button, hold_duration_ms)
                time.sleep(delay_between_ms / 1000)  # Convert ms to seconds
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to press button {button}: {e}")
                success = False
        
        return success
    
    def read_memory(self, address: int, size: int, domain: str = "wram") -> bytes:
        """
        Read a block of memory from the game.
        
        Args:
            address: Memory address to read from (relative to domain base)
            size: Number of bytes to read
            domain: Memory domain to read from
            
        Returns:
            bytes: Bytes read from memory
        """
        # Check if the domain exists in our cached domains
        if domain not in self.memory_domains:
            logger.warning(f"Memory domain {domain} not found in cached domains")
        
        response = self.session.get(
            f"{self.base_url}/memorydomain/readrange", 
            params={
                "memoryDomain": domain,
                "address": f"0x{address:x}",
                "length": f"{size}"
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.content
    
    def read_byte(self, address: int, domain: str = "wram") -> int:
        """
        Read a single byte from memory.
        
        Args:
            address: Memory address to read from (relative to domain base)
            domain: Memory domain to read from
            
        Returns:
            int: The byte value
        """
        response = self.session.get(
            f"{self.base_url}/memorydomain/read8", 
            params={
                "memoryDomain": domain,
                "address": f"0x{address:x}"
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return int(response.text)
    
    def read_word(self, address: int, domain: str = "wram") -> int:
        """
        Read a 16-bit word from memory.
        
        Args:
            address: Memory address to read from (relative to domain base)
            domain: Memory domain to read from
            
        Returns:
            int: The 16-bit value
        """
        response = self.session.get(
            f"{self.base_url}/memorydomain/read16", 
            params={
                "memoryDomain": domain,
                "address": f"0x{address:x}"
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return int(response.text)
    
    def read_dword(self, address: int, domain: str = "wram") -> int:
        """
        Read a 32-bit double word from memory.
        
        Args:
            address: Memory address to read from (relative to domain base)
            domain: Memory domain to read from
            
        Returns:
            int: The 32-bit value
        """
        response = self.session.get(
            f"{self.base_url}/memorydomain/read32", 
            params={
                "memoryDomain": domain,
                "address": f"0x{address:x}"
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return int(response.text)
    
    def write_memory(self, address: int, data: List[int], domain: str = "wram") -> bool:
        """
        Write bytes to a memory address.
        
        Args:
            address: Memory address to write to (relative to domain base)
            data: List of bytes to write
            domain: Memory domain to write to
            
        Returns:
            bool: True if successful
        """
        # Note: we need to call write8, write16, or write32 for each byte/word/dword 
        for i, value in enumerate(data):
            response = self.session.post(
                f"{self.base_url}/memorydomain/write8",
                params={
                    "memoryDomain": domain,
                    "address": f"0x{address + i:x}",
                    "value": f"{value}"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
        return True
    
    def write_byte(self, address: int, value: int, domain: str = "wram") -> bool:
        """
        Write a single byte to memory.
        
        Args:
            address: Memory address to write to (relative to domain base)
            value: Byte value to write (0-255)
            domain: Memory domain to write to
            
        Returns:
            bool: True if successful
        """
        response = self.session.post(
            f"{self.base_url}/memorydomain/write8",
            params={
                "memoryDomain": domain,
                "address": f"0x{address:x}",
                "value": f"{value}"
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return True
    
    def write_word(self, address: int, value: int, domain: str = "wram") -> bool:
        """
        Write a 16-bit word to memory.
        
        Args:
            address: Memory address to write to (relative to domain base)
            value: 16-bit value to write
            domain: Memory domain to write to
            
        Returns:
            bool: True if successful
        """
        response = self.session.post(
            f"{self.base_url}/memorydomain/write16",
            params={
                "memoryDomain": domain,
                "address": f"0x{address:x}",
                "value": f"{value}"
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return True
    
    def write_dword(self, address: int, value: int, domain: str = "wram") -> bool:
        """
        Write a 32-bit double word to memory.
        
        Args:
            address: Memory address to write to (relative to domain base)
            value: 32-bit value to write
            domain: Memory domain to write to
            
        Returns:
            bool: True if successful
        """
        response = self.session.post(
            f"{self.base_url}/memorydomain/write32",
            params={
                "memoryDomain": domain,
                "address": f"0x{address:x}",
                "value": f"{value}"
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return True

    def save_state(self, slot: int = 1) -> bool:
        """
        Save the current state to a slot.
        
        Args:
            slot: Save state slot number (1-9)
            
        Returns:
            bool: True if successful
        """
        response = self.session.post(
            f"{self.base_url}/savestate/save", 
            params={"slot": slot},
            timeout=self.timeout
        )
        response.raise_for_status()
        return True
    
    def load_state(self, slot: int = 1) -> bool:
        """
        Load a state from a slot.
        
        Args:
            slot: Save state slot number (1-9)
            
        Returns:
            bool: True if successful
        """
        response = self.session.post(
            f"{self.base_url}/savestate/load", 
            params={"slot": slot},
            timeout=self.timeout
        )
        response.raise_for_status()
        return True
    
    def reset_game(self) -> bool:
        """
        Reset the game.
        
        Returns:
            bool: True if successful
        """
        response = self.session.post(f"{self.base_url}/coreadapter/reset", timeout=self.timeout)
        response.raise_for_status()
        return True
    
    def log_message(self, message: str, level: str = "info") -> bool:
        """
        Log a message to the console.
        
        Args:
            message: The message to log
            level: The log level (info, error, warn)
            
        Returns:
            bool: True if successful
        """
        if level.lower() == "error":
            endpoint = "error"
        elif level.lower() == "warn":
            endpoint = "warn"
        else:
            endpoint = "info"
            
        response = self.session.post(
            f"{self.base_url}/console/{endpoint}", 
            params={"message": message},
            timeout=self.timeout
        )
        response.raise_for_status()
        return True
    
    # Pokemon-specific methods 
    def get_pokemon_party_count(self) -> int:
        """
        Get the number of Pokemon in the party for the current game.
        
        Returns:
            int: Number of Pokemon in the party
        """
        if self.game_title.startswith("POKEMON EMER"):
            # Pokemon Emerald
            # Party count is at 0x0244e9 in WRAM (relative to base)
            return self.read_byte(0x0244e9)
        else:
            logger.warning(f"Unsupported game for party count: {self.game_title}")
            return 0
    
    def get_pokemon_party(self) -> List[Dict[str, Any]]:
        """
        Get data for all Pokemon in the party for the current game.
        
        Returns:
            List[Dict]: List of Pokemon data dictionaries
        """
        party = []
        count = self.get_pokemon_party_count()
        
        if count == 0:
            return party
            
        if self.game_title.startswith("POKEMON EMER"):
            # Pokemon Emerald
            # Party data starts at 0x0244ec in WRAM (relative to base)
            PARTY_DATA_ADDR = 0x0244ec
            PARTY_MON_SIZE = 100
            
            for i in range(count):
                addr = PARTY_DATA_ADDR + (i * PARTY_MON_SIZE)
                data = self.read_memory(addr, PARTY_MON_SIZE)
                
                # Parse the Pokemon data (simplified)
                personality = int.from_bytes(data[0:4], byteorder='little')
                ot_id = int.from_bytes(data[4:8], byteorder='little')
                
                # Status is at offset 80
                status = int.from_bytes(data[80:84], byteorder='little')
                level = data[84]
                hp = int.from_bytes(data[86:88], byteorder='little')
                max_hp = int.from_bytes(data[88:90], byteorder='little')
                attack = int.from_bytes(data[90:92], byteorder='little')
                defense = int.from_bytes(data[92:94], byteorder='little')
                speed = int.from_bytes(data[94:96], byteorder='little')
                
                pokemon = {
                    'index': i,
                    'personality': personality,
                    'ot_id': ot_id,
                    'level': level,
                    'hp': hp,
                    'max_hp': max_hp,
                    'attack': attack,
                    'defense': defense,
                    'speed': speed,
                    'status': status
                }
                
                party.append(pokemon)
                
        else:
            logger.warning(f"Unsupported game for party data: {self.game_title}")
            
        return party

    def execute_action(self, action_name: str) -> bool:
        """
        Execute a higher-level game action.
        
        These are common sequences of button presses for frequent actions.
        
        Args:
            action_name: The name of the action to execute
            
        Returns:
            bool: True if successful
        """
        actions = {
            "start_game": [Button.START],
            "open_menu": [Button.START],
            "close_menu": [Button.B],
            "advance_text": [Button.A],
            "select_option": [Button.A],
            "cancel": [Button.B],
            "move_up": [Button.UP],
            "move_down": [Button.DOWN],
            "move_left": [Button.LEFT],
            "move_right": [Button.RIGHT],
            "run": [Button.B],  # Hold B while moving in Pokemon
        }
        
        if action_name not in actions:
            logger.error(f"Unknown action: {action_name}")
            raise ValueError(f"Unknown action: {action_name}. "
                           f"Available actions: {list(actions.keys())}")
        
        return self.press_sequence(actions[action_name])


# Usage example
if __name__ == "__main__":
    try:
        # Initialize the controller
        controller = MGBAController(timeout=3)  # Shorter timeout for testing
        
        # Get the game information
        print(f"Game Title: {controller.game_title}")
        print(f"Game Code: {controller.game_code}")
        
        # Get memory domain information
        print("\nMemory Domain Information:")
        for domain, info in controller.memory_domains.items():
            print(f"  {domain}: base=0x{info['base']:x}, size=0x{info['size']:x}")
        
        # Get Pokemon party data if this is a Pokemon game
        if controller.game_title.startswith("POKEMON"):
            print("\nPokemon Party:")
            party_count = controller.get_pokemon_party_count()
            print(f"  Party Count: {party_count}")
            
            if party_count > 0:
                party = controller.get_pokemon_party()
                for pokemon in party:
                    print(f"  Pokemon {pokemon['index']+1}:")
                    print(f"    Level: {pokemon['level']}")
                    print(f"    HP: {pokemon['hp']}/{pokemon['max_hp']}")
                    print(f"    Attack: {pokemon['attack']}")
                    print(f"    Defense: {pokemon['defense']}")
                    print(f"    Speed: {pokemon['speed']}")
        
    except Exception as e:
        print(f"Error: {e}") 